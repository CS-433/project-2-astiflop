import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.train_utils.pipeline_configs import (
    attach_latest_checkpoints,
    load_wrappers_from_config,
)

from models.cnn_attention_models.regression_wrappers import RegressorVisualizationWrapper
from utils.train_utils.dataset import LPBSDataset


def load_dataset(pytorch_dir, scaler_config_path, scaler_type="standard", device="cpu"):
    """
    Load dataset once for caching/efficiency.
    """
    print(f"Loading dataset from {pytorch_dir} on device {device}...")
    dataset = LPBSDataset(
        pytorch_dir,
        scaler_type=scaler_type,
        mode="test",
        scaler_config_path=scaler_config_path,
        device=device,
    )
    return dataset



def run_models_inference(dataset, loaded_models, random_idx=None):
    """
    Run inference on a single common sample for all pre-loaded models.
    """
    if random_idx is None:
        random_idx = random.randint(0, len(dataset) - 1)

    data_tensor, label, total_segments = dataset[random_idx]
    T_actual = int(total_segments.item())

    print(f"Selected sample {random_idx} with {T_actual} valid segments.")

    results = {}

    true_remaining = []
    true_objective = []

    knee_point = 150 // 3  # 50 segments
    for t in range(1, T_actual + 1):
        true_remaining.append(float(T_actual - t))
        true_objective.append(min(float(T_actual - t), knee_point))

     for model_name, wrapper in loaded_models.items():
        print(f"Running inference for {model_name}...")
        preds, vars, custom_data = wrapper.get_trajectory_predictions(
            data_tensor, T_actual
        )

        results[model_name] = {
            "predictions": np.array(preds),
            "variances": np.array(vars),
            "custom_data": custom_data,
        }

    return (
        T_actual,
        np.array(true_objective),
        np.array(true_remaining),
        data_tensor.detach().cpu().numpy(),
        results,
    )


def plot_interactive_results(
    T_actual, true_objective, true_remaining, data_tensor, results_dict
):
    """
    Spawns an interactive matplotlib window showing predictions, trajectories, and attentions.
    """
    steps = np.arange(1, T_actual + 1)

    # Determine bounds
    max_predicted_total = 0
    y_max_val = np.max(true_remaining) * 1.1
    for m_res in results_dict.values():
        preds = m_res["predictions"]
        vars_ = m_res["variances"]
        stds = np.sqrt(np.maximum(0, vars_))
        max_predicted_total = max(max_predicted_total, np.max(steps + preds))
        y_max_val = max(y_max_val, np.max(preds + stds) * 1.1)

    max_true_total = np.max(steps + true_remaining)
    plot1_xlim = max(1.5 * T_actual, max_predicted_total, max_true_total)

    # Set up figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.20, wspace=0.2)
    ax1, ax2 = axs

    # PLOT 1: Timeline
    (line_true,) = ax1.plot(
        [],
        [],
        label="True Remaining",
        linestyle="--",
        marker="o",
        markersize=4,
        color="blue",
        alpha=0.7,
    )
    vline_true = ax1.axvline(
        x=-1, color="blue", linestyle="-", alpha=0.6, label="Current True Total"
    )

    model_lines = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for idx, (model_name, m_res) in enumerate(results_dict.items()):
        color = colors[idx]
        (line,) = ax1.plot(
            [],
            [],
            label=f"{model_name} predictions",
            color=color,
            marker="x",
            markersize=4,
        )
        vline = ax1.axvline(x=-1, color=color, linestyle="--", alpha=0.6)
        model_lines[model_name] = {
            "line": line,
            "vline": vline,
            "color": color,
            "fill": None,
        }

    ax1.set_xlim(0, plot1_xlim)
    ax1.set_ylim(0, y_max_val)
    ax1.set_xlabel("Time (Segments observed)")
    ax1.set_ylabel("Remaining segments")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title("Lifespan Prediction Timeline")

    # PLOT 2: Trajectories
    valid_points_mask = ~(
        (data_tensor[:T_actual, 0, :] == 0)
        & (data_tensor[:T_actual, 1, :] == 0)
        & (data_tensor[:T_actual, 2, :] == 0)
    )
    valid_x = data_tensor[:T_actual, 0, :][valid_points_mask]
    valid_y = data_tensor[:T_actual, 1, :][valid_points_mask]
    if len(valid_x) > 0:
        g_min_x, g_max_x = np.min(valid_x), np.max(valid_x)
        g_min_y, g_max_y = np.min(valid_y), np.max(valid_y)
        dx = (g_max_x - g_min_x) * 0.05
        dy = (g_max_y - g_min_y) * 0.05
        plot2_xlim = (g_min_x - dx, g_max_x + dx)
        plot2_ylim = (g_min_y - dy, g_max_y + dy)
    else:
        plot2_xlim = (-1, 1)
        plot2_ylim = (-1, 1)

    cmap = plt.cm.plasma
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.03, shrink=0.5, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["t=0", f"t={T_actual}"])

    traj_lines = []
    for i in range(T_actual):
        x_data = data_tensor[i, 0, :]
        y_data = data_tensor[i, 1, :]
        valid_mask = ~((x_data == 0) & (y_data == 0))
        if np.any(valid_mask):
            last_valid = np.max(np.nonzero(valid_mask)[0])
            x_filtered = x_data[: last_valid + 1]
            y_filtered = y_data[: last_valid + 1]
        else:
            x_filtered = x_data
            y_filtered = y_data

        (line,) = ax2.plot(
            x_filtered, y_filtered, linewidth=1.5, alpha=0.9, visible=False
        )
        traj_lines.append(line)

    ax2.set_title("C. elegans Trajectory")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.set_xlim(*plot2_xlim)
    ax2.set_ylim(*plot2_ylim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # UI Controls
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Segment (t)", 1, T_actual, valinit=T_actual, valstep=1)

    for i in np.arange(1, T_actual + 1):
        ax_slider.axvline(i, color="black", linewidth=0.8, alpha=0.4, zorder=1)
    ax_slider.set_xticks([])

    ax_lifetime = plt.axes([0.15, 0.015, 0.65, 0.03])
    ax_lifetime.axis("off")
    txt_elapsed = ax_lifetime.text(
        0.0,
        0.5,
        "",
        transform=ax_lifetime.transAxes,
        ha="left",
        va="center",
        fontsize=10,
    )
    txt_remaining = ax_lifetime.text(
        1.0,
        0.5,
        "",
        transform=ax_lifetime.transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )

    def update(val):
        t = int(slider.val)
        current_idx = t - 1

        # Update Timeline
        slider.valtext.set_text(f"{t}")

        line_true.set_data(steps[:t], true_remaining[:t])
        current_true_total = t + true_remaining[current_idx]
        vline_true.set_xdata([current_true_total, current_true_total])

        for model_name, m_res in results_dict.items():
            preds = m_res["predictions"]
            vars_ = m_res["variances"]
            stds = np.sqrt(np.maximum(0, vars_))

            m_lines = model_lines[model_name]
            m_lines["line"].set_data(steps[:t], preds[:t])
            current_pred_total = t + preds[current_idx]
            m_lines["vline"].set_xdata([current_pred_total, current_pred_total])

            if m_lines["fill"] is not None:
                m_lines["fill"].remove()
                m_lines["fill"] = None

            if np.any(stds > 1e-4):
                m_lines["fill"] = ax1.fill_between(
                    steps[:t],
                    preds[:t] - stds[:t],
                    preds[:t] + stds[:t],
                    color=m_lines["color"],
                    alpha=0.2,
                    linewidth=0,
                )

        # Update Trajectories
        for i in range(T_actual):
            if i < t:
                traj_lines[i].set_visible(True)
                if i == t - 1:
                    traj_lines[i].set_color("red")
                    traj_lines[i].set_linewidth(2.5)
                    traj_lines[i].set_zorder(5)
                else:
                    traj_lines[i].set_color(
                        cmap(i / (T_actual - 1)) if T_actual > 1 else cmap(1.0)
                    )
                    traj_lines[i].set_linewidth(1.5)
                    traj_lines[i].set_zorder(2)
            else:
                traj_lines[i].set_visible(False)

        fig.canvas.draw_idle()

    update(T_actual)
    slider.on_changed(update)

    state = {"retry": False}

    ax_btn = plt.axes([0.85, 0.05, 0.1, 0.04])
    btn_retry = Button(ax_btn, "Next Sample", hovercolor="0.975")

    def on_retry(event):
        state["retry"] = True
        plt.close(fig)

    btn_retry.on_clicked(on_retry)

    plt.show()
    return state["retry"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive visualization pipeline for multiple models."
    )
    parser.add_argument(
        "--pytorch_dir",
        "-d",
        type=str,
        default="preprocessed_with_lifespan_test/",
        help="Path to PyTorch preprocessed data directory",
    )
    parser.add_argument(
        "--scaler_config_path",
        "-c",
        type=str,
        default="preprocessed_with_lifespan/scaler_config.json",
        help="Path to the scaler config JSON file",
    )
    parser.add_argument(
        "--scaler",
        "-s",
        type=str,
        default="standard",
        help="Scaler type: 'none', 'minmax', 'standard'",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Explicit dataset index to visualize.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_config = {

    }
    attach_latest_checkpoints(models_config)

    # Adjust paths if executing from project root
    if not os.path.exists(args.scaler_config_path):
        args.scaler_config_path = "preprocessed_with_lifespan/scaler_config.json"
        args.pytorch_dir = "preprocessed_with_lifespan_test/"

    dataset = load_dataset(
        pytorch_dir=args.pytorch_dir,
        scaler_config_path=args.scaler_config_path,
        scaler_type=args.scaler,
        device=device,
    )

    wrappers = load_wrappers_from_config(models_config)

    current_sample_idx = args.sample_idx
    while True:
        T_actual, true_objective, true_remaining, data_tensor, results_dict = (
            run_models_inference(
                dataset=dataset,
                loaded_models=wrappers,
                random_idx=current_sample_idx,
            )
        )

        retry = plot_interactive_results(
            T_actual, true_objective, true_remaining, data_tensor, results_dict
        )

        if not retry:
            break

        # After the first targeted sample, ensuing retries should be random
        current_sample_idx = None
