import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def calculate_average_results(results):
    avg_results = {}
    for model_name, folds in results.items():
        avg_results[model_name] = {}
        for metric in ["acc", "prec", "rec", "f1"]:
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


def plot_results(avg_results):
    metrics = ["acc", "f1"]
    # Sort model names by F1 score in descending order
    model_names = sorted(list(avg_results.keys()), key=lambda x: avg_results[x]["f1"], reverse=True)

    x = np.arange(len(model_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 6))

    def get_color(name):
        if "logReg" in name:
            return "tab:blue"
        elif "rocket" in name:
            return "tab:red"
        elif "tail_mil" in name:
            return "tab:green"
        else:
            return "tab:gray"

    bar_colors = [get_color(name) for name in model_names]

    acc_means = [avg_results[model_name]["acc"] for model_name in model_names]
    acc_stds = [avg_results[model_name]["acc_std"] for model_name in model_names]
    f1_means = [avg_results[model_name]["f1"] for model_name in model_names]
    f1_stds = [avg_results[model_name]["f1_std"] for model_name in model_names]

    rects1 = ax.bar(x - width/2, acc_means, width, yerr=acc_stds, capsize=5, label='Accuracy', color=bar_colors)
    rects2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds, capsize=5, label='F1 Score', color=bar_colors, alpha=0.5)

    ax.bar_label(rects1, padding=3, fmt="%.2f")
    ax.bar_label(rects2, padding=3, fmt="%.2f")

    ax.set_ylabel("Scores")
    ax.set_title("Model Performance: Accuracy vs F1")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    legend_elements = [
        Patch(facecolor='tab:blue', label='logReg'),
        Patch(facecolor='tab:red', label='rocket'),
        Patch(facecolor='tab:green', label='tail_mil'),
        Patch(facecolor='gray', label='Accuracy'),
        Patch(facecolor='gray', alpha=0.5, label='F1 Score'),
    ]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    plt.show()
    plt.savefig("model_performance.png")
    print("Plot saved to model_performance.png")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process and plot model results.")
    parser.add_argument("--results_file", "-r", type=str, default="results.json",
                        help="Path to the JSON file containing model results.")
    args = parser.parse_args()
    # Example usage
    # Load results from a JSON file
    with open(args.results_file, "r") as f:
        avg_results = json.load(f)

    plot_results(avg_results)
