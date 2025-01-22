# Third-party
import torch
import torch_geometric as pyg
from torch import nn
import psutil
import resource
import os
import gc
import time

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

def log_memory_usage(location=""):
    """Log current memory usage with detailed statistics"""
    process = psutil.Process()
    
    # Get memory info
    process_memory = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    # Calculate memory usage
    used_gb = process_memory.rss / (1024 * 1024 * 1024)
    total_gb = system_memory.total / (1024 * 1024 * 1024)
    available_gb = system_memory.available / (1024 * 1024 * 1024)
    
    print(f"\nMemory Usage at {location}:")
    print(f"Process RSS: {used_gb:.2f} GB")
    print(f"System Total: {total_gb:.2f} GB")
    print(f"System Available: {available_gb:.2f} GB")
    print(f"System Used %: {system_memory.percent}%")
    
    # Get tensor memory stats if using PyTorch
    if torch.cuda.is_available():
        print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Get detailed process memory
    print(f"\nDetailed Process Memory:")
    print(f"Virtual Memory: {process_memory.vms / (1024**3):.2f} GB")
    print(f"Shared Memory: {process_memory.shared / (1024**3):.2f} GB")
    print(f"Text Memory: {process_memory.text / (1024**3):.2f} GB")
    print(f"Data Memory: {process_memory.data / (1024**3):.2f} GB")
    
    # Get number of open file descriptors
    print(f"Open Files: {len(process.open_files())}")
    
    # Log PyTorch tensors if debug mode
    if os.getenv("DEBUG_MEMORY"):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    print(f"Found tensor: {obj.size()}, {obj.element_size() * obj.nelement() / 1024**3:.2f} GB")
            except:
                pass

class MemoryTracker:
    """Track detailed memory usage during operations"""
    def __init__(self):
        self.peak_memory = 0
        self.current_operation = None
        self.operation_peaks = {}
        self.tensor_sizes = {}
        self.operation_history = []
        
    def log_tensor(self, name, tensor):
        """Log tensor size and memory usage"""
        if torch.is_tensor(tensor):
            size_bytes = tensor.element_size() * tensor.nelement()
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            device = str(tensor.device)
            self.tensor_sizes[name] = {
                'size_gb': size_bytes / (1024**3),
                'shape': shape,
                'dtype': dtype,
                'device': device
            }
            
    def start_operation(self, name):
        """Start tracking a new operation with detailed memory stats"""
        self.current_operation = name
        self.tensor_sizes.clear()  # Reset tensor tracking for new operation
        
        # Get baseline memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.operation_history.append({
            'operation': name,
            'start_time': time.time(),
            'start_memory': memory_info.rss / (1024**3),
            'tensors': {}
        })
        
    def end_operation(self):
        """End current operation and log detailed stats"""
        if self.current_operation is None:
            return
            
        process = psutil.Process()
        memory_info = process.memory_info()
        current_memory = memory_info.rss / (1024**3)
        
        # Update operation history
        for entry in reversed(self.operation_history):
            if entry['operation'] == self.current_operation:
                entry.update({
                    'end_time': time.time(),
                    'end_memory': current_memory,
                    'memory_change': current_memory - entry['start_memory'],
                    'tensors': self.tensor_sizes.copy()
                })
                break
                
        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            
        # Store peak for operation
        if self.current_operation not in self.operation_peaks:
            self.operation_peaks[self.current_operation] = []
        self.operation_peaks[self.current_operation].append(current_memory)
        
        self.current_operation = None
        
    def report(self):
        """Generate detailed memory usage report"""
        print("\n=== Memory Usage Report ===")
        print(f"Peak Memory Usage: {self.peak_memory:.2f} GB")
        print("\nOperation History:")
        
        for entry in self.operation_history:
            duration = entry.get('end_time', time.time()) - entry['start_time']
            memory_change = entry.get('memory_change', 0)
            print(f"\nOperation: {entry['operation']}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Memory Change: {memory_change:.2f} GB")
            
            if 'tensors' in entry:
                print("Tensor Details:")
                for tensor_name, tensor_info in entry['tensors'].items():
                    print(f"  {tensor_name}:")
                    print(f"    Size: {tensor_info['size_gb']:.2f} GB")
                    print(f"    Shape: {tensor_info['shape']}")
                    print(f"    Type: {tensor_info['dtype']}")
                    print(f"    Device: {tensor_info['device']}")
                    
        print("\nPeak Memory by Operation:")
        for op, peaks in self.operation_peaks.items():
            avg_peak = sum(peaks) / len(peaks)
            max_peak = max(peaks)
            print(f"{op}:")
            print(f"  Average Peak: {avg_peak:.2f} GB")
            print(f"  Max Peak: {max_peak:.2f} GB")
            print(f"  Call Count: {len(peaks)}")

# Create global memory tracker
memory_tracker = MemoryTracker()

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
        memory_tracker.start_operation("forward_pass")
        
        # Log input tensor sizes
        memory_tracker.log_tensor("send_rep", send_rep)
        memory_tracker.log_tensor("rec_rep", rec_rep)
        memory_tracker.log_tensor("edge_rep", edge_rep)
        
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
            memory_tracker.end_operation()
            return rec_rep, edge_rep

        memory_tracker.end_operation()
        return rec_rep

    def message(self, x_j, x_i, edge_attr):
        """
        Compute messages from node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    def aggregate(self, inputs, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        
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

# Local
from . import utils
