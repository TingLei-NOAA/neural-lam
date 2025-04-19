# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import matplotlib.pyplot as plt
from . import config

"""
This script visualizes the statistics computed by create_parameter_weights.py.

How to run:
-----------
python -m neural_lam.plot_parameter_stats \
    --static_dir path/to/stats/directory \
    --data_config neural_lam/data_config.yaml \
    --output_dir stats_plots \
    --prefix your_prefix_if_any

Arguments:
----------
--static_dir : str
    Directory containing the parameter statistics files (required)
--data_config : str
    Path to data config file (default: neural_lam/data_config.yaml)
--output_dir : str
    Directory to save plots (default: stats_plots)
--prefix : str
    Prefix used in the parameter files (default: '')

Input Files:
-----------
The script expects the following files in static_dir:
1. {prefix}mean.pt - Tensor of mean values for each feature
2. {prefix}std.pt - Tensor of standard deviation values for each feature
3. flux_stats.pt - (Optional) Tensor of flux statistics

Output:
-------
Creates three visualization files:
1. {prefix}feature_stats.png
   - Bar plots showing mean and standard deviation for each feature
2. {prefix}feature_distributions.png
   - Distribution range plot showing mean ±2σ for each feature
3. flux_stats.png (if flux statistics exist)
   - Bar plot of flux mean and standard deviation
"""

def plot_feature_stats(mean, std, var_names, output_dir, prefix=""):
    """Plot mean and standard deviation for each feature"""
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot means
    x = np.arange(len(var_names))
    ax1.bar(x, mean.numpy())
    ax1.set_xticks(x)
    ax1.set_xticklabels(var_names, rotation=45, ha='right')
    ax1.set_title('Mean Values')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot standard deviations
    ax2.bar(x, std.numpy())
    ax2.set_xticks(x)
    ax2.set_xticklabels(var_names, rotation=45, ha='right')
    ax2.set_title('Standard Deviations')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}feature_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_distributions(mean, std, var_names, output_dir, prefix=""):
    """Plot distribution ranges (mean ± 2*std) for each feature"""
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(var_names))
    
    # Plot mean points
    plt.plot(x, mean.numpy(), 'o', label='Mean')
    
    # Plot error bars (±2 standard deviations)
    plt.errorbar(x, mean.numpy(), yerr=2*std.numpy(), fmt='none', capsize=5, 
                label='±2σ Range', alpha=0.5)
    
    plt.xticks(x, var_names, rotation=45, ha='right')
    plt.title('Feature Distributions (Mean ± 2σ)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_flux_stats(flux_stats, output_dir):
    """Plot flux statistics"""
    
    flux_mean, flux_std = flux_stats
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a bar plot with mean and std
    values = [flux_mean.item(), flux_std.item()]
    labels = ['Mean', 'Standard Deviation']
    
    ax.bar(labels, values)
    ax.set_title('Flux Statistics')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flux_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = ArgumentParser(description="Plot parameter statistics")
    parser.add_argument(
        "--static_dir",
        type=str,
        required=True,
        help="Directory containing the parameter statistics files"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stats_plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix used in the parameter files (default: '')"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config to get variable names
    config_loader = config.Config.from_file(args.data_config)
   #cltorg  var_names = config_loader.data_vars
    var_names = config_loader.dataset.var_names
    
    # Load statistics
    mean_path = os.path.join(args.static_dir, f"{args.prefix}mean.pt")
    std_path = os.path.join(args.static_dir, f"{args.prefix}std.pt")
    flux_path = os.path.join(args.static_dir, "flux_stats.pt")
    
    mean = torch.load(mean_path)
    std = torch.load(std_path)
    
    # Create plots for feature statistics
    plot_feature_stats(mean, std, var_names, args.output_dir, args.prefix)
    plot_feature_distributions(mean, std, var_names, args.output_dir, args.prefix)
    
    # If flux statistics exist, plot them
    if os.path.exists(flux_path):
        flux_stats = torch.load(flux_path)
        plot_flux_stats(flux_stats, args.output_dir)
    
    print(f"Plots have been saved to: {args.output_dir}")
    print(f"Generated plots:")
    print(f"1. {args.prefix}feature_stats.png - Bar plots of means and standard deviations")
    print(f"2. {args.prefix}feature_distributions.png - Distribution ranges for each feature")
    if os.path.exists(flux_path):
        print("3. flux_stats.png - Flux statistics")

if __name__ == "__main__":
    main()
