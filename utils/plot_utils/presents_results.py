import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def calculate_average_results(results):
    avg_results = {}
    for model_name, folds in results.items():
        avg_results[model_name] = {}
        for metric in folds[next(iter(folds))].keys():  # Get metric names from the first fold
            values = [f[metric] for f in folds.values()]
            avg_results[model_name][metric] = np.mean(values)
            avg_results[model_name][f"{metric}_std"] = np.std(values)
    return avg_results


def save_results_to_json(results, filename="results.json"):
    # Convert numpy types to python types for json serialization
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(filename, "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=4)
    print(f"Results saved to {filename}")


def plot_results(avg_results, save_path="model_performance.png"):
    # Sort model names by first metric in descending order
    model_names = sorted(list(avg_results.keys()), key=lambda x: list(avg_results[x].values())[0], reverse=True)

    # Extract metrics (exclude *_std fields)
    all_keys = list(avg_results[model_names[0]].keys())
    metrics = [k for k in all_keys if not k.endswith('_std')]

    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(14, 6))

    def get_color(metric_index):
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
        return colors[metric_index % len(colors)]

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        means = [avg_results[model_name][metric] for model_name in model_names]
        stds = [avg_results[model_name].get(f"{metric}_std", 0) for model_name in model_names]
        offset = (i - len(metrics)/2 + 0.5) * width
        rects = ax.bar(x + offset, means, width, yerr=stds, capsize=5, label=metric.capitalize(), color=get_color(i), alpha=0.8)
        ax.bar_label(rects, padding=3, fmt="%.2f")

    ax.set_ylabel("Scores")
    ax.set_title("Models average performances across folds")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_cnn_comparison(results_summary, save_path="cnn_model_comparison.png"):
    """
    Plots the F1 comparison of different CNN models.
    
    Args:
        results_summary (dict): Dictionary where keys are model names and values are dicts containing 'f1_mean' and 'f1_std'.
        save_path (str): Path to save the plot.
    """
    model_names = list(results_summary.keys())
    f1_means = [results_summary[m]["f1_mean"] for m in model_names]
    f1_stds = [results_summary[m]["f1_std"] for m in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, f1_means, yerr=f1_stds, capsize=5, color='skyblue', edgecolor='black', alpha=0.8)
    
    plt.ylabel('F1 Score')
    plt.title('Model Comparison (F1 Score)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar, mean_val in zip(bars, f1_means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to '{save_path}'")
    # plt.show()
