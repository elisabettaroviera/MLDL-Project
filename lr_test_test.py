# visualize_lr_range_npz.py

import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_lr_curve(file_path, apply_smoothing=True):
    data = np.load(file_path)
    lrs = data['lrs']
    losses = data['losses']
    smoothed_losses = data['smoothed_losses'] if 'smoothed_losses' in data.files else losses

    print(f"Loaded {file_path}")
    print(f"Min raw loss: {np.min(losses):.6f} at LR: {lrs[np.argmin(losses)]:.6f}")
    print(f"Min smoothed loss: {np.min(smoothed_losses):.6f} at LR: {lrs[np.argmin(smoothed_losses)]:.6f}")
    print(f"LR range: [{lrs[0]:.2e}, {lrs[-1]:.2e}]")

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, label='Raw Loss', color='gray', alpha=0.5)
    if apply_smoothing:
        plt.plot(lrs, smoothed_losses, label='Smoothed Loss', color='blue')
    plt.xscale('log')
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss")
    plt.title("LR Range Test - Loss vs Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="lr_range_data_pidnet.npz", help="Path to .npz file")
    parser.add_argument('--no-smooth', action='store_true', help="Disable smoothing display")
    args = parser.parse_args()

    plot_lr_curve(args.file, apply_smoothing=not args.no_smooth)
