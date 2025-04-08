import matplotlib.pyplot as plt
import numpy as np
import re

def parse_losses(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract training and validation losses using regex
    train_match = re.search(r'Training losses: \[(.*?)\]', content)
    val_match = re.search(r'Validation losses: \[(.*?)\]', content)
    
    if train_match and val_match:
        # Clean and parse training losses
        train_losses = train_match.group(1)
        train_losses = re.sub(r'0:\s*', '', train_losses)  # Remove SLURM prefixes
        train_losses = [float(x.strip()) for x in train_losses.split(',') if x.strip()]
        
        # Clean and parse validation losses
        val_losses = val_match.group(1)
        val_losses = re.sub(r'0:\s*', '', val_losses)  # Remove SLURM prefixes
        val_losses = [float(x.strip()) for x in val_losses.split(',') if x.strip()]
        
        return train_losses, val_losses
    return None, None

def plot_losses(train_losses, val_losses, output_file='training_curves.png'):
    plt.figure(figsize=(12, 6))
    epochs = range(len(train_losses))
    
    # Plot training loss
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.7)
    
    # Plot validation loss
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', color='red', alpha=0.7)
    
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add moving average for smoother visualization
    window = 20
    train_ma = np.convolve(train_losses, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(train_losses)), train_ma, 
             color='darkblue', alpha=0.5, linestyle='--', 
             label=f'Training {window}-epoch Moving Average')
    
    if val_losses:
        val_ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(val_losses)), val_ma, 
                color='darkred', alpha=0.5, linestyle='--', 
                label=f'Validation {window}-epoch Moving Average')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    train_losses, val_losses = parse_losses("d0")  # Change this to your SLURM output file
    if train_losses and val_losses:
        plot_losses(train_losses, val_losses)
        
        # Print some statistics
        print("\nTraining Loss Statistics:")
        print(f"Initial: {train_losses[0]:.4f}")
        print(f"Final: {train_losses[-1]:.4f}")
        print(f"Best: {min(train_losses):.4f}")
        print(f"Mean: {np.mean(train_losses):.4f}")
        
        print("\nValidation Loss Statistics:")
        print(f"Initial: {val_losses[0]:.4f}")
        print(f"Final: {val_losses[-1]:.4f}")
        print(f"Best: {min(val_losses):.4f}")
        print(f"Mean: {np.mean(val_losses):.4f}")
    else:
        print("Could not parse losses from the file")
