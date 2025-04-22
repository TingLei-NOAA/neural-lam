# Standard library
import json
import random
import time
from argparse import ArgumentParser
import os
import numpy as np

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Local
from . import WeatherDataset, config, utils
from .models import GraphLAM, HiLAM, HiLAMParallel

torch.set_default_dtype(torch.float32)

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


def main(input_args=None):
    """
    Main function for training and evaluating models with Distributed Data Parallel (DDP) support
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM with DDP support"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        type=int,
        default=0,
        help="Train only on control member of ensemble data "
        "(default: 0 (False))",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="adam option beta: first one (default: 0.9)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.95, help="adam option beta: second one (default: 0.95)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight for L2 regularization of weights (default: 0.0)"
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="accumulated number of batches for use of gradients (default: 1)"
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )

    # Logger Settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural_lam",
        help="Wandb project name (default: neural_lam)",
    )
    parser.add_argument(
        "--val_steps_to_log",
        type=list,
        default=[1, 2, 3],
        help="Steps to log val loss for (default: [1, 2, 3])",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )

    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }
    config_loader = config.Config.from_file(args.data_config)

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Get SLURM info for DDP
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    world_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    # Load data with DDP samplers
    train_dataset = WeatherDataset(
        config_loader.dataset.name,
        pred_length=args.ar_steps,
        split="train",
        subsample_step=args.step_length,
        subset=bool(args.subset_ds),
        control_only=args.control_only,
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=world_rank,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.n_workers,
        pin_memory=True
    )

    max_pred_length = (19 // args.step_length) - 2
    val_dataset = WeatherDataset(
        config_loader.dataset.name,
        pred_length=max_pred_length,
        split="val",
        subsample_step=args.step_length,
        subset=bool(args.subset_ds),
        control_only=args.control_only,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=world_rank,
        shuffle=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.n_workers,
        pin_memory=True
    )

    # Set up device
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    # Create model parameters
    model_class = MODELS[args.model]
    model = model_class(args)

    # Set up run name and logging
    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )
    print("Run name:", run_name)

    # Set up checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_last=True,
    )

    # Set up logging
    logger = pl.loggers.WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=args,
    )

    class LossTracker(pl.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.train_losses = []
            self.val_losses = []
            self.epochs = []
            
        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            train_loss = trainer.callback_metrics.get('train_loss')
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
                
            self.epochs.append(epoch)
            self.train_losses.append(train_loss)
                
        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get('val_mean_loss')
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            self.val_losses.append(val_loss)
            
        def get_losses(self):
            return {
                'epochs': self.epochs,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }

    loss_tracker = LossTracker()

    # Initialize trainer with DDP strategy
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",  # Explicitly use DDP
        accelerator=device_name,
        devices=ntasks_per_node,
        num_nodes=num_nodes,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, loss_tracker],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        sync_batchnorm=True  # Important for DDP
    )

    # Initialize wandb metrics on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger, args.val_steps_to_log)

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            test_dataset = WeatherDataset(
                config_loader.dataset.name,
                pred_length=max_pred_length,
                split="test",
                subsample_step=args.step_length,
                subset=bool(args.subset_ds),
            )
            
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=world_rank,
                shuffle=False
            )

            eval_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                sampler=test_sampler,
                num_workers=args.n_workers,
                pin_memory=True
            )

        print(f"Running evaluation on {args.eval}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        model.to(device)
        model = model.float()
        trainer.test(model=model, dataloaders=eval_loader, ckpt_path=args.load)

    else:
        # Train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        model.to(device)
        model = model.float()

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.load,
        )

    # After training
    completed_epochs = trainer.current_epoch
    
    @rank_zero_only
    def print_losses():
        print(f"Training completed after {completed_epochs} epochs.")
        losses = loss_tracker.get_losses()
        print("Epochs:", loss_tracker.epochs)
        np.savetxt("training_losses.txt", loss_tracker.train_losses)
        np.savetxt("validation_losses.txt", loss_tracker.val_losses)
    
    print_losses()


if __name__ == "__main__":
    main()
