# Third-party
import torch
import torch_geometric as pyg
from torch import nn
import psutil
import resource
import os
import gc

def check_system_memory(location=""):
    """Check system and process memory usage"""
    mem = psutil.virtual_memory()
    print(f"\nMemory Check at {location}:")
    print(f"Total system memory: {mem.total / (1024**3):.2f} GB")
    print(f"Available memory: {mem.available / (1024**3):.2f} GB")
    print(f"Used memory: {mem.used / (1024**3):.2f} GB")
    
    # Get process memory info
    process = psutil.Process()
    print(f"Process memory usage: {process.memory_info().rss / (1024**3):.2f} GB")
    
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Process memory limits - Soft: {'unlimited' if soft == -1 else f'{soft/(1024**3):.2f} GB'}, "
              f"Hard: {'unlimited' if hard == -1 else f'{hard/(1024**3):.2f} GB'}")
    except Exception as e:
        print(f"Could not get process limits: {e}")

def check_memory_fragmentation():
    """Check memory fragmentation status"""
    gc.collect()
    
    # Get memory maps
    maps_file = f"/proc/{os.getpid()}/maps"
    if os.path.exists(maps_file):
        with open(maps_file, 'r') as f:
            memory_maps = f.readlines()
            
        # Analyze contiguous regions
        regions = []
        for line in memory_maps:
            if 'heap' in line or 'anon' in line:
                addr_range = line.split()[0]
                start, end = [int(x, 16) for x in addr_range.split('-')]
                regions.append(end - start)
                
        if regions:
            largest_block = max(regions)
            print(f"Largest contiguous memory block: {largest_block / (1024**2):.2f} MB")

def defragment_memory():
    """Attempt to defragment memory by forcing allocation and deallocation"""
    gc.collect()
    
    # Get current memory usage
    process = psutil.Process()
    current_mem = process.memory_info().rss
    
    # Allocate and immediately free a large block to consolidate memory
    try:
        temp = torch.empty(int(current_mem * 1.2), dtype=torch.uint8)
        del temp
    except:
        pass
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Local
from . import utils


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

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
        """
        Create a new InteractionNet

        edge_index: (2,M), Edges in pyg format
        input_dim: Dimensionality of input representations,
            for both nodes and edges
        update_edges: If new edge representations should be computed
            and returned
        hidden_layers: Number of hidden layers in MLPs
        hidden_dim: Dimensionality of hidden layers, if None then same
            as input_dim
        edge_chunk_sizes: List of chunks sizes to split edge representation
            into and use separate MLPs for (None = no chunking, same MLP)
        aggr_chunk_sizes: List of chunks sizes to split aggregated node
            representation into and use separate MLPs for
            (None = no chunking, same MLP)
        aggr: Message aggregation method (sum/mean)
        """
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr)

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
        """
        Apply interaction network to update the representations of receiver
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
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep
        )
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connections
        rec_rep = rec_rep + rec_diff

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message(self, x_i, x_j, edge_attr):
        """Compute messages from node j to node i."""
        check_system_memory("Before message concatenation")
        print(f"x_i shape: {x_i.shape}, memory: {x_i.element_size() * x_i.nelement() / 1024 / 1024:.2f}MB")
        print(f"x_j shape: {x_j.shape}, memory: {x_j.element_size() * x_j.nelement() / 1024 / 1024:.2f}MB")
        print(f"edge_attr shape: {edge_attr.shape}, memory: {edge_attr.element_size() * edge_attr.nelement() / 1024 / 1024:.2f}MB")
        total_concat_memory = (x_i.element_size() * x_i.nelement() + 
                             x_j.element_size() * x_j.nelement() + 
                             edge_attr.element_size() * edge_attr.nelement()) / 1024 / 1024
        print(f"Total memory for concatenation: {total_concat_memory:.2f}MB")
        
        # Check memory fragmentation before operation
        check_memory_fragmentation()
        
        # Try to defragment memory
        defragment_memory()
        
        try:
            # Pre-allocate output tensor with correct dimensions
            # Keep batch dimension and node dimension, concatenate along feature dimension
            batch_size, num_nodes, feat_dim = x_i.shape
            concat_tensor = torch.empty(
                (batch_size, num_nodes, feat_dim * 3),
                dtype=x_i.dtype, 
                device=x_i.device, 
                pin_memory=False
            )
            
            # Copy data into pre-allocated tensor
            concat_tensor[:, :, :feat_dim] = edge_attr
            concat_tensor[:, :, feat_dim:2*feat_dim] = x_j
            concat_tensor[:, :, 2*feat_dim:] = x_i
            
            result = self.edge_mlp(concat_tensor)
            check_system_memory("After message concatenation")
            return result
        except Exception as e:
            check_system_memory("After message concatenation ERROR")
            check_memory_fragmentation()  # Check fragmentation after error
            raise e

    # pylint: disable-next=signature-differs
    def aggregate(self, inputs, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
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
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)
