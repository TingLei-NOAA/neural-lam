# Third-party
import torch
import torch_geometric as pyg
from torch import nn

# Local
from . import utils


class InteractionNet(pyg.nn.MessagePassing):
    """Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    def __init__(
        self,
        edge_index,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
    ):
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr, flow="source_to_target")

        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim

        # Make both sender and receiver indices of edge_index start at 0
        edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]
        # Store number of receiver nodes according to edge_index
        self.num_rec = edge_index[1].max() + 1
        edge_index[0] = (
            edge_index[0] + self.num_rec
        )  # Make sender indices after rec
        self.register_buffer("edge_index", edge_index, persistent=False)

        # Create MLPs
        edge_mlp_recipe = [3 * input_dim] + [hidden_dim] * (hidden_layers + 1)
        aggr_mlp_recipe = [2 * input_dim] + [hidden_dim] * (hidden_layers + 1)

        if edge_chunk_sizes is None:
            self.edge_mlp = utils.make_mlp(edge_mlp_recipe)
        else:
            self.edge_mlp = SplitMLPs(
                [utils.make_mlp(edge_mlp_recipe) for _ in edge_chunk_sizes],
                edge_chunk_sizes,
            )

        if aggr_chunk_sizes is None:
            self.aggr_mlp = utils.make_mlp(aggr_mlp_recipe)
        else:
            self.aggr_mlp = SplitMLPs(
                [utils.make_mlp(aggr_mlp_recipe) for _ in aggr_chunk_sizes],
                aggr_chunk_sizes,
            )

        self.update_edges = update_edges

    def forward(self, send_rep, rec_rep, edge_rep):
        """Apply interaction network to update the representations of receiver
        nodes, and optionally the edge representations.

        send_rep: (N_send, d_h), vector representations of sender nodes
        rec_rep: (N_rec, d_h), vector representations of receiver nodes
        edge_rep: (M, d_h), vector representations of edges used

        Returns:
        rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
        (optionally) edge_rep: (M, d_h), updated vector representations
            of edges
        """
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)

        # Use memory efficient propagation
        edge_rep_aggr = self.propagate(
            self.edge_index,
            x=node_reps,
            edge_attr=edge_rep,
            size=(node_reps.size(-2), self.num_rec)
        )

        # Free memory
        del node_reps
        torch.cuda.empty_cache()

        # Update node features
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))
        rec_rep = rec_rep + rec_diff

        if self.update_edges:
            # Update edge features - do this in chunks to save memory
            chunk_size = 100000  # Adjust this based on available memory
            num_edges = edge_rep.size(0)
            edge_chunks = []
            
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                chunk_idx = slice(i, end_idx)
                
                # Process edge updates in chunks
                edge_inputs = torch.cat((
                    edge_rep[chunk_idx],
                    send_rep[self.edge_index[0][chunk_idx]],
                    rec_rep[self.edge_index[1][chunk_idx]]
                ), dim=-1)
                
                edge_diff_chunk = self.edge_mlp(edge_inputs)
                edge_chunks.append(edge_rep[chunk_idx] + edge_diff_chunk)
                
                # Free memory after each chunk
                del edge_inputs, edge_diff_chunk
                torch.cuda.empty_cache()
            
            # Combine chunks
            edge_rep = torch.cat(edge_chunks, dim=0)
            return rec_rep, edge_rep
            
        return rec_rep

    def message(self, x_j, x_i, edge_attr):
        """
        Compute messages from node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate messages to receiver nodes.
        """
        return super().aggregate(inputs, index, ptr=ptr, dim_size=self.num_rec)


class SplitMLPs(nn.Module):
    """Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x):
        """
        Chunk up input and feed through MLPs

        x: (..., N, d), where N = sum(chunk_sizes)

        Returns:
        joined_output: (..., N, d), concatenated results from the MLPs
        """
        # Split input into chunks
        chunks = x.split(self.chunk_sizes, dim=-2)
        # Apply MLPs
        outputs = [mlp(chunk) for mlp, chunk in zip(self.mlps, chunks)]
        # Join outputs
        return torch.cat(outputs, dim=-2)
