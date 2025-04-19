import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

def moving_average(data, window_size=20):
    # Use uniform_filter1d for moving average, with edge handling
    weights = np.ones(window_size) / window_size
    return uniform_filter1d(data, size=window_size, mode='reflect')

def plot_losses(input_file, output_file='loss_plot.png'):
    # Read the losses from the file
    train_losses = []
    val_losses = []
    
    current_list = None
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "# Training losses":
                current_list = train_losses
            elif line == "# Validation losses":
                current_list = val_losses
            elif line and not line.startswith('#'):
                current_list.append(float(line))

    # Create epoch indices
    epochs = np.arange(len(train_losses))

    # Calculate moving averages
    train_ma = moving_average(train_losses)
    val_ma = moving_average(val_losses)

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw data with lower alpha
    plt.plot(epochs, train_losses, label='Training Loss', color='#2ecc71', alpha=0.3, linewidth=1)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#e74c3c', alpha=0.3, linewidth=1)
    
    # Plot moving averages with solid lines
    plt.plot(epochs, train_ma, label='Training Loss (20-epoch MA)', 
             color='#2ecc71', linewidth=2)
    plt.plot(epochs, val_ma, label='Validation Loss (20-epoch MA)', 
             color='#e74c3c', linewidth=2)

    # Customize the plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Losses with 20-epoch Moving Average', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Set y-axis to log scale since the losses vary widely
    plt.yscale('log')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

    # Print some statistics
    print("\nLoss Statistics:")
    print(f"Training Loss - Final: {train_losses[-1]:.6f}, Min: {min(train_losses):.6f}, Max: {max(train_losses):.6f}")
    print(f"Validation Loss - Final: {val_losses[-1]:.6f}, Min: {min(val_losses):.6f}, Max: {max(val_losses):.6f}")

if __name__ == "__main__":
    plot_losses("losses.txt", "loss_plot.png")
