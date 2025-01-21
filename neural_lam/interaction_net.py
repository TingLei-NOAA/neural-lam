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

def cleanup_memory():
    """Clean up memory aggressively"""
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Synchronize CUDA to ensure memory is freed
        torch.cuda.synchronize()
    
    # Try to release memory back to OS
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

class MemoryTracker:
    """Track peak memory usage during operations"""
    def __init__(self):
        self.peak_memory = 0
        self.current_operation = None
        self.operation_peaks = {}

    def start_operation(self, name):
        """Start tracking memory for an operation"""
        self.current_operation = name
        process = psutil.Process()
        self.operation_peaks[name] = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"\nStarting operation: {name}")
        print(f"Initial memory: {self.operation_peaks[name]:.2f} MB")

    def end_operation(self):
        """End tracking memory for current operation"""
        if self.current_operation:
            process = psutil.Process()
            current_mem = process.memory_info().rss / (1024 * 1024)  # MB
            peak = max(current_mem, self.operation_peaks[self.current_operation])
            self.operation_peaks[self.current_operation] = peak
            print(f"\nEnding operation: {self.current_operation}")
            print(f"Peak memory during operation: {peak:.2f} MB")
            self.current_operation = None

    def report(self):
        """Report peak memory usage for all operations"""
        print("\nMemory Usage Report:")
        print("-" * 50)
        for op, peak in self.operation_peaks.items():
            print(f"{op}: {peak:.2f} MB")
        print("-" * 50)

# Create global memory tracker
memory_tracker = MemoryTracker()

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
        memory_tracker.start_operation("forward_pass")
        cleanup_memory()  # Clean before major operation
        print("Starting forward pass")
        check_system_memory("Before forward pass")
        
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep
        )
        cleanup_memory()  # Clean after propagation
        
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))
        cleanup_memory()  # Clean after aggregation
        
        # Residual connections
        rec_rep = rec_rep + rec_diff
        cleanup_memory()  # Clean after update
        
        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message(self, x_i, x_j, edge_attr):
        """Compute messages from node j to node i."""
        # Ensure inputs are on CPU
        x_i = x_i.cpu()
        x_j = x_j.cpu()
        edge_attr = edge_attr.cpu()
        
        memory_tracker.start_operation("message_function")
        check_system_memory("Before message concatenation")
        print(f"x_i shape: {x_i.shape}, memory: {x_i.element_size() * x_i.nelement() / 1024 / 1024:.2f}MB")
        print(f"x_j shape: {x_j.shape}, memory: {x_j.element_size() * x_j.nelement() / 1024 / 1024:.2f}MB")
        print(f"edge_attr shape: {edge_attr.shape}, memory: {edge_attr.element_size() * edge_attr.nelement() / 1024 / 1024:.2f}MB")
        total_concat_memory = (x_i.element_size() * x_i.nelement() + 
                             x_j.element_size() * x_j.nelement() + 
                             edge_attr.element_size() * edge_attr.nelement()) / 1024 / 1024
        print(f"Total memory for concatenation: {total_concat_memory:.2f}MB")
        
        try:
            memory_tracker.start_operation("chunk_processing")
            # Process in smaller chunks
            batch_size, num_nodes, feat_dim = x_i.shape
            chunk_size = 500  # Reduced chunk size for lower memory usage
            results = []
            
            print(f"Processing {num_nodes} nodes in chunks of {chunk_size}")
            
            for start_idx in range(0, num_nodes, chunk_size):
                end_idx = min(start_idx + chunk_size, num_nodes)
                print(f"Processing chunk {start_idx}-{end_idx} ({end_idx-start_idx} nodes)")
                
                # Process chunk with explicit memory management
                with torch.no_grad():  # Disable gradients for concatenation
                    chunk_input = torch.cat([
                        edge_attr[:, start_idx:end_idx],
                        x_j[:, start_idx:end_idx],
                        x_i[:, start_idx:end_idx]
                    ], dim=-1).cpu()  # Ensure on CPU
                
                # Process chunk
                chunk_result = self.edge_mlp(chunk_input).cpu()  # Ensure on CPU
                results.append(chunk_result)
                
                # Force cleanup every chunk
                cleanup_memory()
                
                # Clean up intermediate tensors
                del chunk_input
                if len(results) > 1:
                    # Keep only the concatenated results
                    results = [torch.cat(results, dim=1).cpu()]  # Ensure on CPU
                
                if len(results) % 2 == 0:  # More frequent cleanup
                    print(f"Cleaning up memory after {len(results)} chunks")
                    check_system_memory(f"After processing {len(results)} chunks")
            
            memory_tracker.end_operation()  # End chunk processing
            
            memory_tracker.start_operation("final_concatenation")
            print(f"Concatenating final results")
            
            # Final concatenation
            result = torch.cat(results, dim=1).cpu()  # Ensure on CPU
            cleanup_memory()
            check_system_memory("After final concatenation")
            memory_tracker.end_operation()
            
            memory_tracker.end_operation()  # End message function
            memory_tracker.report()
            return result
            
        except Exception as e:
            check_system_memory("After message concatenation ERROR")
            print(f"Error during message computation: {str(e)}")
            memory_tracker.end_operation()
            memory_tracker.report()
            raise e

    # pylint: disable-next=signature-differs
    def aggregate(self, inputs, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        # Ensure inputs are on CPU
        inputs = inputs.cpu()
        index = index.cpu() if index is not None else None
        ptr = ptr.cpu() if ptr is not None else None
        
        cleanup_memory()  # Clean before aggregation
        
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
        # Ensure inputs are on CPU
        x = x.cpu()
        
        cleanup_memory()  # Clean before chunking
        
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        cleanup_memory()  # Clean after chunk processing
        
        return torch.cat(chunk_outputs, dim=-2)
