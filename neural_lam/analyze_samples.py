# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local
from . import WeatherDataset, config

"""
This script analyzes the statistics of weather dataset variables and creates histograms.

How to run:
-----------
python -m neural_lam.analyze_samples \
    --data_config neural_lam/data_config.yaml \
    --split train \
    --output_dir stats_output \
    --pred_length 19 \
    --step_length 1

Arguments:
----------
--data_config : str
    Path to data config file (default: neural_lam/data_config.yaml)
--split : str
    Dataset split to analyze (choices: train, val, test) (default: train)
--output_dir : str
    Directory to save output plots (default: stats_output)
--pred_length : int
    Prediction length for dataset (default: 19)
--step_length : int
    Step length for dataset (default: 1)

Output:
-------
For each variable in the dataset:
1. Prints statistics (min, max, mean, std, median) to console
2. Creates a histogram plot with statistics in a text box
3. Saves plot as '{variable_name}_distribution.png' in the output directory
"""

def analyze_dataset(dataset, config_loader, output_dir):
    """
    Analyze dataset statistics and create histograms
    
    Args:
        dataset: WeatherDataset instance
        config_loader: Config instance with dataset information
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics containers
    num_vars = config_loader.num_data_vars()
    all_values = [[] for _ in range(num_vars)]
    var_names = config_loader.data_vars
    
    # Collect all values
    print("Collecting statistics...")
    for idx in tqdm(range(len(dataset))):
        init_states, target_states, _ = dataset[idx]
        # Combine init and target states for analysis
        all_states = torch.cat([init_states, target_states], dim=0)
        
        # For each variable
        for var_idx in range(num_vars):
            values = all_states[..., var_idx].flatten()
            all_values[var_idx].extend(values.cpu().numpy())
    
    # Calculate and plot statistics
    print("\nCalculating statistics and creating plots...")
    for var_idx in range(num_vars):
        var_values = np.array(all_values[var_idx])
        var_name = var_names[var_idx]
        
        # Calculate statistics
        stats = {
            'min': np.min(var_values),
            'max': np.max(var_values),
            'mean': np.mean(var_values),
            'std': np.std(var_values),
            'median': np.median(var_values)
        }
        
        # Print statistics
        print(f"\nStatistics for {var_name}:")
        for stat_name, stat_value in stats.items():
            print(f"{stat_name}: {stat_value:.6f}")
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(var_values, bins=50, density=True, alpha=0.7)
        plt.title(f'Distribution of {var_name}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # Add statistics text box
        stats_text = '\n'.join([f'{k}: {v:.6f}' for k, v in stats.items()])
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{var_name}_distribution.png'))
        plt.close()

def main():
    """
    Main function to analyze dataset statistics
    """
    parser = ArgumentParser(
        description="Analyze statistics of weather dataset variables"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stats_output",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--pred_length",
        type=int,
        default=19,
        help="Prediction length for dataset"
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length for dataset"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_loader = config.Config.from_file(args.data_config)
    
    # Create dataset
    dataset = WeatherDataset(
        config_loader.dataset.name,
        pred_length=args.pred_length,
        split=args.split,
        subsample_step=args.step_length
    )
    
    # Analyze dataset
    analyze_dataset(dataset, config_loader, args.output_dir)
    
    print(f"\nAnalysis complete! Plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
