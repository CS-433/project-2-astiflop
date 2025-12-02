import json
import numpy as np
import matplotlib.pyplot as plt


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
    metrics = ["acc", "prec", "rec", "f1"]
    model_names = list(avg_results.keys())

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        means = [avg_results[model_name][m] for m in metrics]
        stds = [avg_results[model_name][f"{m}_std"] for m in metrics]

        offset = width * i - width * (len(model_names) - 1) / 2
        rects = ax.bar(x + offset, means, width, yerr=stds, capsize=5, label=model_name)
        ax.bar_label(rects, padding=3, fmt="%.2f")

    ax.set_ylabel("Scores")
    ax.set_title("Average Model Performance by Metric (with Error Bars)")
    ax.set_xticks(x, metrics)
    ax.legend()

    fig.tight_layout()
    plt.show()
    plt.savefig("model_performance.png")
    print("Plot saved to model_performance.png")
    plt.close()


if __name__ == "__main__":
    # Example usage
    # Load results from a JSON file
    with open("avg_results.json", "r") as f:
        avg_results = json.load(f)

    plot_results(avg_results)
