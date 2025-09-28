# Standard library
import os
import subprocess
from argparse import ArgumentParser
import psutil

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
        sample=self.base_dataset[
            self.original_indices[-1]
            if idx >= self.total_samples
            else idx % len(self.base_dataset)
        ]
        return sample
    def __getitem__(self, idx):
        return self.base_dataset[
            self.original_indices[-1]
            if idx >= self.total_samples
            else idx % len(self.base_dataset)
        ]




        print(f"Sample {idx} has shape: {sample.shape}")
#clt        return self.base_dataset[
#clt            self.original_indices[-1]
#clt            if idx >= self.total_samples
#clt            else idx % len(self.base_dataset)

    def __len__(self):
        return self.total_samples + self.padded_samples

    def get_original_indices(self):
        return self.original_indices

    def get_original_window_indices(self, step_length):
        return [
            i // step_length
            for i in range(len(self.original_indices) * step_length)
        ]


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
    print("thinkdeb save_stats static_dir_path ",static_dir_path) 
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


def process_batch_efficiently(batch, device, delete_after_use=True):
    """Process a single batch efficiently with memory cleanup"""
    init_batch, target_batch, forcing_batch = batch
    
    # Move to device one at a time
    init_batch = init_batch.to(device)
    target_batch = target_batch.to(device)
    forcing_batch = forcing_batch.to(device)
    
    # Process in smaller chunks
    batch = torch.cat((init_batch, target_batch), dim=1)
    flux_batch = forcing_batch[:, :, :, 1]
    
    # Calculate statistics
    means = torch.mean(batch, dim=(1, 2))
    squares = torch.mean(batch**2, dim=(1, 2))
    flux_means = torch.mean(flux_batch)
    flux_squares = torch.mean(flux_batch**2)
    
    # Move results to CPU and convert to numpy to free GPU memory
    means = means.cpu()
    squares = squares.cpu()
    flux_means = flux_means.cpu()
    flux_squares = flux_squares.cpu()
    
    if delete_after_use:
        # Clean up GPU memory
        del init_batch, target_batch, forcing_batch, batch, flux_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return means, squares, flux_means, flux_squares


def process_in_subbatches(batch, sub_batch_size=4):
    """Process a large batch in smaller chunks"""
    init_batch, target_batch, forcing_batch = batch
    total_size = init_batch.size(0)
    
    sub_means = []
    sub_squares = []
    sub_flux_means = []
    sub_flux_squares = []
    
    for i in range(0, total_size, sub_batch_size):
        end_idx = min(i + sub_batch_size, total_size)
        sub_init = init_batch[i:end_idx]
        sub_target = target_batch[i:end_idx]
        sub_forcing = forcing_batch[i:end_idx]
        
        # Process sub-batch
        sub_batch = torch.cat((sub_init, sub_target), dim=1)
        sub_flux = sub_forcing[:, :, :, 1]
        
        # Calculate statistics
        sub_means.append(torch.mean(sub_batch, dim=(1, 2)))
        sub_squares.append(torch.mean(sub_batch**2, dim=(1, 2)))
        sub_flux_means.append(torch.mean(sub_flux))
        sub_flux_squares.append(torch.mean(sub_flux**2))
        
        # Clean up
        del sub_batch, sub_flux
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine results
    means = torch.cat(sub_means, dim=0)
    squares = torch.cat(sub_squares, dim=0)
    flux_means = torch.mean(torch.tensor(sub_flux_means))
    flux_squares = torch.mean(torch.tensor(sub_flux_squares))
    
    return means, squares, flux_means, flux_squares


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


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
        default=16,
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
        help="Number of workers in data loader (default: 0)",
    )
    parser.add_argument(
        "--distributed",
        type=int,
        default=0,
        help="Run the script in distributed mode (1) or not (0) (default: 0)",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Enable memory efficient processing",
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
        # Remove the cuda.set_device call since we're forcing CPU

    if rank == 0:
        static_dir_path = os.path.join(
            "data", config_loader.dataset.name, "static"
        )
        # Create parameter weights based on height
        # based on fig A.1 in graph cast paper
#clt        w_dict = {
#clt            "2": 1.0,
#clt            "0": 0.1,
#clt            "65": 0.065,
#clt            "1000": 0.1,
#clt            "850": 0.05,
#clt            "500": 0.03,
#clt        }

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
#clt        pred_length=63,
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
        num_workers=min(args.n_workers, 2),
        sampler=sampler,
        pin_memory=False,
    )

    if rank == 0:
        print("Computing mean and std.-dev. for parameters...")
    means, squares, flux_means, flux_squares = [], [], [], []

    print("thinkdeb 2") 
    # Process the batch
    for batch in tqdm(loader):
        if rank == 0:
            print_memory_usage()
        
        if distributed:
            init_batch, target_batch, forcing_batch = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
        
        # Free memory after processing
        batch_means, batch_squares, batch_flux_means, batch_flux_squares = process_in_subbatches(
            batch, sub_batch_size=4
        )
        
        means.append(batch_means)
        squares.append(batch_squares)
        flux_means.append(batch_flux_means)
        flux_squares.append(batch_flux_squares)
        
        if rank == 0:
            print_memory_usage()
        
        # Optional: Clear some memory after each batch
        if args.memory_efficient:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if distributed and world_size > 1:
        means_gathered, squares_gathered = [None] * world_size, [
            None
        ] * world_size
        # Gather parameter stats (per-sample tensors)
        dist.all_gather_object(means_gathered, torch.cat(means, dim=0))
        dist.all_gather_object(squares_gathered, torch.cat(squares, dim=0))

        # For flux stats, aggregate globally via all-reduce on sums and counts.
        # These were collected as per-batch means, so we match that semantics
        # by averaging these per-batch means uniformly across all ranks/batches.
        if torch.cuda.is_available():
            red_device = torch.device(f"cuda:{rank}")
        else:
            red_device = torch.device("cpu")

        local_flux_means = torch.tensor(flux_means, dtype=torch.float32, device=red_device) if len(flux_means) > 0 else torch.tensor([], dtype=torch.float32, device=red_device)
        local_flux_squares = torch.tensor(flux_squares, dtype=torch.float32, device=red_device) if len(flux_squares) > 0 else torch.tensor([], dtype=torch.float32, device=red_device)

        local_count = torch.tensor([float(local_flux_means.numel())], dtype=torch.float32, device=red_device)
        local_sum = torch.tensor([float(local_flux_means.sum().item() if local_flux_means.numel() > 0 else 0.0)], dtype=torch.float32, device=red_device)
        local_sq_sum = torch.tensor([float(local_flux_squares.sum().item() if local_flux_squares.numel() > 0 else 0.0)], dtype=torch.float32, device=red_device)

        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)

        if rank == 0:
            means_gathered, squares_gathered = torch.cat(
                means_gathered, dim=0
            ), torch.cat(squares_gathered, dim=0)

            original_indices = ds.get_original_indices()
            means, squares = [means_gathered[i] for i in original_indices], [
                squares_gathered[i] for i in original_indices
            ]

            # Compute global flux mean and second moment from reduced sums
            # Note: if local_count is zero (shouldn't happen), avoid div by zero
            if local_count.item() > 0:
                global_flux_mean = (local_sum / local_count).squeeze(0).cpu()
                global_flux_sq_mean = (local_sq_sum / local_count).squeeze(0).cpu()
                flux_means = [global_flux_mean]
                flux_squares = [global_flux_sq_mean]
            else:
                flux_means, flux_squares = [], []
    else:
        print("thinkdeb 8") 
        means = [torch.cat(means, dim=0)]  # (N_batch, d_features,)
        squares = [torch.cat(squares, dim=0)]  # (N_batch, d_features,)
        flux_means = [torch.tensor(flux_means)]  # (N_batch,)
        flux_squares = [torch.tensor(flux_squares)]  # (N_batch,)
        print("thinkdeb 9 rank ",rank) 
        print("static_dir_path  ",static_dir_path) 

    if rank == 0:
        save_stats(
            static_dir_path,
            means,
            squares,
            flux_means,
            flux_squares,
            "parameter",
        )

    if distributed:
        dist.barrier()

    if rank == 0:
        print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
#clt        pred_length=63,
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
        num_workers=min(args.n_workers, 2),
        sampler=sampler_standard,
        pin_memory=False,
    )
#clt    used_subsample_len = (65 // args.step_length) * args.step_length
    used_subsample_len = (19 // args.step_length) * args.step_length

    diff_means, diff_squares = [], []

    for init_batch, target_batch, _ in tqdm(loader_standard, disable=rank != 0):
        if distributed:
            init_batch, target_batch = init_batch.to(device), target_batch.to(
                device
            )
        # (N_batch, N_t', N_grid, d_features)
        batch = torch.cat((init_batch, target_batch), dim=1)
        print ("thinkdeb init_batch.sahep ",init_batch.shape,' ',target_batch.shape)
        print("thinkdeb batch.shape ",batch.shape)
        print("thinkdeb 3 ", " ",used_subsample_len," ",args.step_length)
        print("thinkdeb 3 ", [ss_i for ss_i in range(args.step_length)])
        # Note: batch contains only 1h-steps
        stepped_batch = torch.cat(
            [
                batch[:, ss_i : used_subsample_len : args.step_length]
                for ss_i in range(args.step_length)
            ],
            dim=0,
        )
        # (N_batch', N_t, N_grid, d_features),
        # N_batch' = args.step_length*N_batch
        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
        # (N_batch', N_t-1, N_grid, d_features)
        diff_means.append(torch.mean(batch_diffs, dim=(1, 2)).cpu())
        # (N_batch', d_features,)
        diff_squares.append(torch.mean(batch_diffs**2, dim=(1, 2)).cpu())
        # (N_batch', d_features,)
        print("thinkdeb 10") 

    if distributed and world_size > 1:
        dist.barrier()
        diff_means_gathered, diff_squares_gathered = [None] * world_size, [
            None
        ] * world_size
        dist.all_gather_object(
            diff_means_gathered, torch.cat(diff_means, dim=0)
        )
        dist.all_gather_object(
            diff_squares_gathered, torch.cat(diff_squares, dim=0)
        )

        if rank == 0:
            diff_means_gathered, diff_squares_gathered = torch.cat(
                diff_means_gathered, dim=0
            ).view(-1, *diff_means[0].shape), torch.cat(
                diff_squares_gathered, dim=0
            ).view(
                -1, *diff_squares[0].shape
            )
            original_indices = ds_standard.get_original_window_indices(
                args.step_length
            )
            diff_means, diff_squares = [
                diff_means_gathered[i] for i in original_indices
            ], [diff_squares_gathered[i] for i in original_indices]

    diff_means = [torch.cat(diff_means, dim=0)]  # (N_batch', d_features,)
    diff_squares = [torch.cat(diff_squares, dim=0)]  # (N_batch', d_features,)
    print("thinkdeb 11 rank",rank) 

    if rank == 0:
        save_stats(static_dir_path, diff_means, diff_squares, [], [], "diff")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
