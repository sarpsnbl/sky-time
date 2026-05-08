"""
plot_comparison.py
==================
Reads the JSONL logs from two models and produces an overlay plot of their
validation MAE curves averaged across folds.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from visualize_training import load_log

def get_mean_curves(log_path):
    """Load log and return epochs and mean val MAE across folds."""
    epoch_records, _ = load_log(log_path)
    if not epoch_records:
        return [], []
    
    # Group by epoch
    epochs = sorted(list({r["epoch"] for r in epoch_records}))
    mean_val_mae = []
    
    for ep in epochs:
        recs = [r for r in epoch_records if r["epoch"] == ep]
        val_maes = [r["val_mae"] for r in recs]
        mean_val_mae.append(np.mean(val_maes))
        
    return epochs, mean_val_mae

def main():
    parser = argparse.ArgumentParser(description="Compare validation curves of two models.")
    parser.add_argument("--model1_log", type=str, default="checkpoints/train_log_convnext_tiny.jsonl",
                        help="Path to the first model's log file.")
    parser.add_argument("--model2_log", type=str, default="checkpoints/train_log_swin_t.jsonl",
                        help="Path to the second model's log file.")
    parser.add_argument("--model1_label", type=str, default="ConvNeXt Tiny",
                        help="Label for the first model in the legend.")
    parser.add_argument("--model2_label", type=str, default="Swin Transformer T",
                        help="Label for the second model in the legend.")
    parser.add_argument("--out", type=str, default="figures/model_comparison.png",
                        help="Path to save the output figure.")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))
    
    curves_plotted = 0
    
    if os.path.exists(args.model1_log):
        ep1, val1 = get_mean_curves(args.model1_log)
        if ep1:
            plt.plot(ep1, val1, label=args.model1_label, color="#4C8EDA", linewidth=2.5)
            curves_plotted += 1
        else:
            print(f"Warning: No valid data in {args.model1_log}")
    else:
        print(f"Warning: {args.model1_log} not found.")
        
    if os.path.exists(args.model2_log):
        ep2, val2 = get_mean_curves(args.model2_log)
        if ep2:
            plt.plot(ep2, val2, label=args.model2_label, color="#e64a19", linewidth=2.5)
            curves_plotted += 1
        else:
            print(f"Warning: No valid data in {args.model2_log}")
    else:
        print(f"Warning: {args.model2_log} not found.")
        
    if curves_plotted == 0:
        print("Error: No logs found to plot. Please check the paths.")
        return
        
    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Validation MAE (minutes)", fontsize=12, fontweight="bold")
    plt.title("Model Comparison: Validation MAE", fontsize=14, fontweight="bold", pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Use aesthetic improvements
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot to {args.out}")

if __name__ == "__main__":
    main()
