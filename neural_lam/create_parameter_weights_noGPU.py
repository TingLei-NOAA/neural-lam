# Standard library
import os
import subprocess
from argparse import ArgumentParser
import psutil
import gc

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Local
from . import WeatherDataset, config


class PaddedWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, world_size, batch_size):
        super().__init__()
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_samples = len(base_dataset)
        self.padded_samples = (
            (self.world_size * self.batch_size) - self.total_samples
        ) % self.world_size
        self.original_indices = list(range(len(base_dataset)))
        self.padded_indices = list(
            range(self.total_samples, self.total_samples + self.padded_samples)
        )

    def __getitem__(self, idx):
        sample = self.base_dataset[
            self.original_indices[-1]
            if idx >= self.total_samples
            else idx % len(self.base_dataset)
        ]
        return sample

    def __len__(self):
        return self.total_samples + self.padded_samples

    def get_original_indices(self):
        return self.original_indices

    def get_original_window_indices(self, step_length):
        return [
            i // step_length
            for i in range(len(self.original_indices) * step_length)
        ]


class RunningStats:
    """Calculate running mean and variance using Welford's online algorithm"""
    def __init__(self, device="cpu"):
        self.n = 0
        self.mean = 0
        self.M2 = 0
        self.device = device

    def update(self, batch):
        """Update stats with new batch"""
        batch_mean = torch.mean(batch, dim=(1, 2))  # Mean over time and grid dimensions
        batch_size = batch_mean.size(0)
        
        # Update counts
        self.n += batch_size
        
        # Update mean and variance using Welford's online algorithm
        delta = batch_mean - self.mean
        self.mean += (delta * batch_size).sum(dim=0) / self.n
        delta2 = batch_mean - self.mean
        self.M2 += (delta * delta2).sum(dim=0)
    
    def get_stats(self):
        """Get current mean and std"""
        variance = self.M2 / (self.n - 1) if self.n > 1 else torch.zeros_like(self.mean)
        std = torch.sqrt(variance)
        return self.mean, std


def get_rank():
    return int(os.environ.get("SLURM_PROCID", 0))


def get_world_size():
    return int(os.environ.get("SLURM_NTASKS", 1))


def setup(rank, world_size):  # pylint: disable=redefined-outer-name
    """Initialize the distributed group."""
    if "SLURM_JOB_NODELIST" in os.environ:
        master_node = (
            subprocess.check_output(
                "scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1",
                shell=True,
            )
            .strip()
            .decode("utf-8")
        )
    else:
        print(
            "\033[91mCareful, you are running this script with --distributed "
            "without any scheduler. In most cases this will result in slower "
            "execution and the --distributed flag should be removed.\033[0m"
        )
        master_node = "localhost"
    os.environ["MASTER_ADDR"] = master_node
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print(
            f"Initialized {dist.get_backend()} "
            f"process group with world size {world_size}."
        )


def save_stats(
    static_dir_path, means, squares, flux_means, flux_squares, filename_prefix
):
    means = (
        torch.stack(means) if len(means) > 1 else means[0]
    )  # (N_batch, d_features,)
    squares = (
        torch.stack(squares) if len(squares) > 1 else squares[0]
    )  # (N_batch, d_features,)
    mean = torch.mean(means, dim=0)  # (d_features,)
    second_moment = torch.mean(squares, dim=0)  # (d_features,)
    std = torch.sqrt(second_moment - mean**2)  # (d_features,)
    print("thinkdeb save_stats static_dir_path ", static_dir_path)
    print("thinkdeb save_stats filename ", f"{filename_prefix}_mean.pt")
    torch.save(
        mean.cpu(), os.path.join(static_dir_path, f"{filename_prefix}_mean.pt")
    )
    torch.save(
        std.cpu(), os.path.join(static_dir_path, f"{filename_prefix}_std.pt")
    )

    if len(flux_means) == 0:
        return
    flux_means = (
        torch.stack(flux_means) if len(flux_means) > 1 else flux_means[0]
    )  # (N_batch,)
    flux_squares = (
        torch.stack(flux_squares) if len(flux_squares) > 1 else flux_squares[0]
    )  # (N_batch,)
    flux_mean = torch.mean(flux_means)  # (,)
    flux_second_moment = torch.mean(flux_squares)  # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
    torch.save(
        torch.stack((flux_mean, flux_std)).cpu(),
        os.path.join(static_dir_path, "flux_stats.pt"),
    )


def print_memory_usage(prefix=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"{prefix} Memory usage: {memory_mb:.2f} MB")


def cleanup_memory():
    """Force garbage collection and clear cuda cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_grid_in_chunks(batch, chunk_size=100, running_stats=None):
    """Process a large grid in smaller chunks"""
    grid_h, grid_w = batch.shape[2:4]
    
    # Process grid in chunks
    for h in range(0, grid_h, chunk_size):
        h_end = min(h + chunk_size, grid_h)
        for w in range(0, grid_w, chunk_size):
            w_end = min(w + chunk_size, grid_w)
            
            # Process this grid chunk
            chunk = batch[:, :, h:h_end, w:w_end, :]
            if running_stats is not None:
                running_stats.update(chunk)
            
            # Clean up
            del chunk
            cleanup_memory()


def process_batch_incrementally(init_batch, target_batch, forcing_batch, running_stats, flux_stats, device, chunk_size=100):
    """Process a single batch updating running statistics"""
    try:
        # Move to device and process
        init_batch = init_batch.to(device)
        target_batch = target_batch.to(device)
        forcing_batch = forcing_batch.to(device)
        
        # Calculate main batch statistics in chunks
        batch = torch.cat((init_batch, target_batch), dim=1)
        process_grid_in_chunks(batch, chunk_size, running_stats)
        
        # Clear main batch memory
        del batch
        cleanup_memory()
        
        # Calculate flux statistics in chunks
        flux_batch = forcing_batch[:, :, :, :, 1]
        process_grid_in_chunks(flux_batch.unsqueeze(-1), chunk_size, flux_stats)
        
        # Clear remaining tensors
        del init_batch, target_batch, forcing_batch, flux_batch
        cleanup_memory()
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        cleanup_memory()
        raise


def process_differences_incrementally(batch, step_length, running_diff_stats):
    """Process differences incrementally for a single batch"""
    used_subsample_len = (batch.size(1) // step_length) * step_length
    
    # Process each step length separately to avoid loading all at once
    for ss_i in range(step_length):
        # Get this slice
        slice_data = batch[:, ss_i:used_subsample_len:step_length]
        
        # Calculate differences between consecutive time steps
        diffs = slice_data[:, 1:] - slice_data[:, :-1]
        
        # Update running statistics
        running_diff_stats.update(diffs)
        
        # Clean up
        del slice_data, diffs
        cleanup_memory()


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step (default: 3)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--distributed",
        type=int,
        default=0,
        help="Run the script in distributed mode (1) or not (0) (default: 0)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Size of grid chunks to process at once",
    )
    args = parser.parse_args()
    distributed = bool(args.distributed)

    rank = get_rank()
    world_size = get_world_size()
    config_loader = config.Config.from_file(args.data_config)

    if distributed:
        setup(rank, world_size)
        device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )
        print(f" device is set as {device} but was forced to cpu")
        device = "cpu"
        torch.cuda.set_device(device) if torch.cuda.is_available() else None

    if rank == 0:
        static_dir_path = os.path.join(
            "data", config_loader.dataset.name, "static"
        )
        print_memory_usage("Initial")

        # Create parameter weights based on height
        # based on fig A.1 in graph cast paper
        w_dict = {
            "2": 1.0,
            "0": 0.1,
            "65": 0.065,
            "1000": 0.1,
            "925": 0.1,
            "850": 0.05,
            "500": 0.03,
            "200": 0.01,
        }

        w_list = np.array(
            [
                w_dict[par.split("_")[-2]]
                for par in config_loader.dataset.var_longnames
            ]
        )
        print("Saving parameter weights...")
        np.save(
            os.path.join(static_dir_path, "parameter_weights.npy"),
            w_list.astype("float32"),
        )
    print("thinkdeb 1")

    # Load dataset without any subsampling
    ds = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=17,
        standardize=False,
    )
    if distributed:
        ds = PaddedWeatherDataset(
            ds,
            world_size,
            args.batch_size,
        )
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(
        ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        sampler=sampler,
    )

    if rank == 0:
        print_memory_usage("After dataset load")

    # Initialize running statistics
    running_stats = RunningStats(device)
    flux_stats = RunningStats(device)

    print("thinkdeb 2")
    # Process the batch
    for init_batch, target_batch, forcing_batch in tqdm(
        loader,
        mininterval=60.0,  # Minimum time between updates (in seconds)
        maxinterval=300.0,  # Maximum time between updates (in seconds)
        disable=rank != 0,  # Only show progress for rank 0
    ):
        if rank == 0:
            print_memory_usage("Before batch processing")
        
        # Process batch incrementally
        process_batch_incrementally(
            init_batch, target_batch, forcing_batch,
            running_stats, flux_stats, device,
            chunk_size=args.chunk_size
        )
        
        if rank == 0:
            print_memory_usage("After batch processing")
        
        cleanup_memory()
    
    if distributed and world_size > 1:
        # Gather statistics from all processes
        if rank == 0:
            print_memory_usage("Before gathering results")
        
        # Get final stats
        mean, std = running_stats.get_stats()
        flux_mean, flux_std = flux_stats.get_stats()
        
        # Save stats
        if rank == 0:
            save_stats(
                static_dir_path,
                [mean.cpu()],
                [std.cpu()],
                [flux_mean.cpu()],
                [flux_std.cpu()],
                "parameter",
            )
            print_memory_usage("After saving stats")
    else:
        # Get final stats
        mean, std = running_stats.get_stats()
        flux_mean, flux_std = flux_stats.get_stats()
        
        # Save stats
        save_stats(
            static_dir_path,
            [mean.cpu()],
            [std.cpu()],
            [flux_mean.cpu()],
            [flux_std.cpu()],
            "parameter",
        )
        print_memory_usage("After saving stats")

    cleanup_memory()

    if rank == 0:
        print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=17,
        standardize=True,
    )  # Re-load with standardization
    if distributed:
        ds_standard = PaddedWeatherDataset(
            ds_standard,
            world_size,
            args.batch_size,
        )
        sampler_standard = DistributedSampler(
            ds_standard, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        sampler_standard = None
    loader_standard = torch.utils.data.DataLoader(
        ds_standard,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        sampler=sampler_standard,
    )
    used_subsample_len = (19 // args.step_length) * args.step_length

    # Initialize running statistics for differences
    running_diff_stats = RunningStats(device)

    for init_batch, target_batch, _ in tqdm(loader_standard, disable=rank != 0):
        if distributed:
            init_batch = init_batch.to(device)
            target_batch = target_batch.to(device)
        
        # Combine batches
        batch = torch.cat((init_batch, target_batch), dim=1)
        
        # Process differences incrementally
        process_differences_incrementally(batch, args.step_length, running_diff_stats)
        
        # Clean up
        del batch, init_batch, target_batch
        cleanup_memory()
    
    # Get final difference statistics
    diff_mean, diff_std = running_diff_stats.get_stats()
    
    if rank == 0:
        save_stats(
            static_dir_path, 
            [diff_mean.cpu()], 
            [diff_std.cpu()], 
            [], [], 
            "diff"
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
