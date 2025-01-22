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
        
        # Ensure inputs are on CPU
        send_rep = send_rep.cpu()
        rec_rep = rec_rep.cpu()
        edge_rep = edge_rep.cpu()
        
        try:
            # Process in chunks to save memory
            chunk_size = 500
            num_nodes = rec_rep.size(0)
            output_chunks = []
            
            for i in range(0, num_nodes, chunk_size):
                memory_tracker.start_operation(f"chunk_processing_{i}")
                
                chunk_end = min(i + chunk_size, num_nodes)
                chunk_mask = (self.edge_index[1] >= i) & (self.edge_index[1] < chunk_end)
                chunk_edge_index = self.edge_index[:, chunk_mask]
                
                # Log chunk sizes
                memory_tracker.log_tensor("chunk_mask", chunk_mask)
                memory_tracker.log_tensor("chunk_edge_index", chunk_edge_index)
                
                # Process chunk
                chunk_out = self._forward_chunk(send_rep, rec_rep[i:chunk_end], chunk_edge_index, edge_rep)
                output_chunks.append(chunk_out)
                
                memory_tracker.end_operation()
            
            # Combine results
            memory_tracker.start_operation("combine_chunks")
            rec_rep = torch.cat(output_chunks, dim=0)
            memory_tracker.log_tensor("combined_output", rec_rep)
            memory_tracker.end_operation()
            
            if self.update_edges:
                memory_tracker.start_operation("update_edges")
                edge_rep = edge_rep + self.edge_mlp(torch.cat((edge_rep, send_rep, rec_rep), dim=-1))
                memory_tracker.log_tensor("updated_edge_rep", edge_rep)
                memory_tracker.end_operation()
                return rec_rep, edge_rep
                
            memory_tracker.end_operation()
            return rec_rep
            
        except Exception as e:
            memory_tracker.end_operation()
            raise e

    def _forward_chunk(self, send_rep, rec_rep, edge_index, edge_rep):
        """Process a single chunk of nodes with memory-efficient operations"""
        memory_tracker.start_operation("forward_chunk")
        
        try:
            # Log input sizes
            memory_tracker.log_tensor("chunk_send_rep", send_rep)
            memory_tracker.log_tensor("chunk_rec_rep", rec_rep)
            memory_tracker.log_tensor("chunk_edge_index", edge_index)
            memory_tracker.log_tensor("chunk_edge_rep", edge_rep)
            
            # Process in smaller sub-chunks
            sub_chunk_size = 50000  # Adjust based on available memory
            num_edges = edge_index.size(1)
            edge_features_list = []
            
            # Get unique node indices and create mapping
            unique_send = torch.unique(edge_index[0])
            unique_rec = torch.unique(edge_index[1])
            
            # Create mappings from global to local indices
            send_mapping = {int(idx): i for i, idx in enumerate(unique_send)}
            rec_mapping = {int(idx): i for i, idx in enumerate(unique_rec)}
            
            # Create local representations
            local_send_rep = send_rep[unique_send]
            local_rec_rep = rec_rep[unique_rec]
            
            for start_idx in range(0, num_edges, sub_chunk_size):
                memory_tracker.start_operation(f"sub_chunk_{start_idx}")
                
                end_idx = min(start_idx + sub_chunk_size, num_edges)
                sub_edge_index = edge_index[:, start_idx:end_idx]
                
                # Map global indices to local indices
                sub_send_nodes = torch.tensor([send_mapping[int(idx)] for idx in sub_edge_index[0]], 
                                           device=sub_edge_index.device)
                sub_rec_nodes = torch.tensor([rec_mapping[int(idx)] for idx in sub_edge_index[1]], 
                                          device=sub_edge_index.device)
                
                # Extract features using local indices
                sub_send_rep = local_send_rep[sub_send_nodes]
                sub_rec_rep = local_rec_rep[sub_rec_nodes]
                
                # Handle edge representation
                if edge_rep.dim() > 1:
                    sub_edge_rep = edge_rep[start_idx:end_idx]
                else:
                    sub_edge_rep = edge_rep
                
                # Compute features for sub-chunk
                with torch.no_grad():  # Temporarily disable grad to save memory
                    sub_input = torch.cat((sub_edge_rep, sub_send_rep, sub_rec_rep), dim=-1)
                    sub_features = self.edge_mlp(sub_input)
                    edge_features_list.append(sub_features)
                
                # Clean up
                del sub_input, sub_features, sub_send_rep, sub_rec_rep, sub_edge_rep
                gc.collect()
                memory_tracker.end_operation()
            
            # Combine features
            memory_tracker.start_operation("combine_features")
            edge_features = torch.cat(edge_features_list, dim=0)
            del edge_features_list
            gc.collect()
            memory_tracker.end_operation()
            
            # Aggregate messages in chunks
            memory_tracker.start_operation("aggregate_messages")
            aggr_chunk_size = 10000  # Adjust based on available memory
            aggr_messages_list = []
            
            for start_idx in range(0, len(unique_rec), aggr_chunk_size):
                end_idx = min(start_idx + aggr_chunk_size, len(unique_rec))
                receiver_subset = unique_rec[start_idx:end_idx]
                
                # Get messages for these receivers
                receiver_mask = torch.isin(edge_index[1], receiver_subset)
                subset_features = edge_features[receiver_mask]
                subset_edge_index = edge_index[:, receiver_mask]
                
                # Map global indices to local for aggregation
                local_edge_index = torch.tensor([rec_mapping[int(idx)] for idx in subset_edge_index[1]], 
                                             device=subset_edge_index.device)
                
                # Aggregate using local indices
                subset_messages = self.aggregate(subset_features, local_edge_index - start_idx, dim=0)
                aggr_messages_list.append(subset_messages)
                
                # Clean up
                del subset_features, subset_edge_index, receiver_mask, local_edge_index
                gc.collect()
            
            # Combine aggregated messages
            aggr_messages = torch.cat(aggr_messages_list, dim=0)
            del aggr_messages_list
            gc.collect()
            memory_tracker.end_operation()
            
            # Final node update
            memory_tracker.start_operation("node_update")
            out = self.aggr_mlp(torch.cat((rec_rep, aggr_messages), dim=-1))
            memory_tracker.end_operation()
            
            memory_tracker.end_operation()
            return out
            
        except Exception as e:
            print(f"Error in _forward_chunk: {str(e)}")
            print(f"Tensor shapes:")
            print(f"send_rep: {send_rep.shape}")
            print(f"rec_rep: {rec_rep.shape}")
            print(f"edge_index: {edge_index.shape}")
            print(f"edge_rep: {edge_rep.shape}")
            memory_tracker.end_operation()
            raise e

    def message(self, x_i, x_j, edge_attr):
        """Compute messages from node j to node i."""
        # Ensure inputs are on CPU
        x_i = x_i.cpu()
        x_j = x_j.cpu()
        edge_attr = edge_attr.cpu()
        
        # Process messages in chunks
        chunk_size = 500
        num_edges = x_i.size(0)
        message_chunks = []
        
        for i in range(0, num_edges, chunk_size):
            chunk_end = min(i + chunk_size, num_edges)
            
            # Process chunk
            chunk_x_i = x_i[i:chunk_end].cpu()
            chunk_x_j = x_j[i:chunk_end].cpu()
            chunk_edge_attr = edge_attr[i:chunk_end].cpu()
            
            # Compute messages for chunk
            chunk_msg = self.edge_mlp(torch.cat((chunk_edge_attr, chunk_x_i, chunk_x_j), dim=-1))
            message_chunks.append(chunk_msg)
            
            # Clean up
            del chunk_x_i, chunk_x_j, chunk_edge_attr, chunk_msg
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Combine results
        return torch.cat(message_chunks, dim=0)

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

# Local
from . import utils
