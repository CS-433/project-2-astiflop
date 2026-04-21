import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import numpy as np

path = './notebook/inference_dump.pt'

dump_dict = torch.load(path, weights_only=False)
T_actual =          int(dump_dict['T_actual'])
true_objective =    np.array(dump_dict['true_objective'])
true_remaining =    np.array(dump_dict['true_remaining'])
predictions =       np.array(dump_dict['predictions'])
s_weights_cpu =     [np.array(x) for x in dump_dict['s_weights_cpu']]
v_weights_cpu =     [np.array(x) for x in dump_dict['v_weights_cpu']]
data_tensor =       np.array(dump_dict['data_tensor'], dtype=np.float32)

# Global variables
steps = np.arange(1, T_actual + 1)
max_predicted_total = np.max(steps + predictions)
max_true_total = np.max(steps + true_remaining)

# Set up figure
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.2)
ax1, ax2 = axs[0]
ax3, ax4 = axs[1]

# ---------------------------------------------------------
# PLOT 1 (Timeline)
# ---------------------------------------------------------
plot1_xlim = max(1.5 * T_actual, max_predicted_total, max_true_total)
y_max_val = max(np.max(true_remaining), np.max(predictions)) * 1.1

line_true, = ax1.plot([], [], label="True Remaining", linestyle="--", marker="o", markersize=4, color="blue", alpha=0.7)
line_pred, = ax1.plot([], [], label="Predicted Remaining", color='red', marker="x", markersize=4)
vline_pred = ax1.axvline(x=-1, color='red', linestyle='-', alpha=0.6, label='Current Pred Total')
vline_true = ax1.axvline(x=-1, color='blue', linestyle='-', alpha=0.6, label='Current True Total')

ax1.set_xlim(0, plot1_xlim)
ax1.set_ylim(0, y_max_val)
ax1.set_xlabel("Time (Segments observed)")
ax1.set_ylabel("Remaining segments")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title("Lifespan Prediction Timeline")

# ---------------------------------------------------------
# PLOT 2 (Trajectories)
# ---------------------------------------------------------
# Constraints
valid_points_mask = ~((data_tensor[:T_actual, 0, :] == 0) & (data_tensor[:T_actual, 1, :] == 0) & (data_tensor[:T_actual, 2, :] == 0))
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

# Define the lines and their colors
cmap = plt.cm.plasma
traj_lines = []
for i in range(T_actual):
    x_data = data_tensor[i, 0, :]
    y_data = data_tensor[i, 1, :]
    valid_mask = ~((x_data == 0) & (y_data == 0) & (data_tensor[i, 2, :] == 0))
    if np.any(valid_mask):
        last_valid = np.max(np.nonzero(valid_mask)[0])
        x_filtered = x_data[:last_valid+1]
        y_filtered = y_data[:last_valid+1]
    else:
        x_filtered = x_data
        y_filtered = y_data
            
    line, = ax2.plot(x_filtered, y_filtered, linewidth=1.5, alpha=0.9, visible=False)
    traj_lines.append(line)


ax2.set_title("C. elegans Trajectory")
ax2.set_xlabel("X coordinate")
ax2.set_ylabel("Y coordinate")
ax2.set_xlim(*plot2_xlim)
ax2.set_ylim(*plot2_ylim)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle='--', alpha=0.3)

# ---------------------------------------------------------
# PLOT 3 (Attention Box)
# ---------------------------------------------------------
true_lifespan = max_true_total
colors = ["#ff9999", "#66b3ff", "#99ff99"]
variates = ["X", "Y", "Speed"]

bar_containers = []
for i in range(3):
    # Initialize with zeros for heights. 
    # We will update these heights directly on update() using `.patches`
    bc = ax3.bar(np.arange(1, T_actual + 1), np.zeros(T_actual), color=colors[i], label=variates[i], edgecolor='black', linewidth=0.5, alpha=0.8)
    bar_containers.append(bc)

ax3.set_xlim(0, true_lifespan)
ax3.set_ylim(0, 0.8)
ax3.set_xlabel("Segment Index")
ax3.set_ylabel("Attention Weight")
ax3.set_title("Current Segment Attention by Feature")
ax3.legend()
ax3.grid(True, axis='y', linestyle='--', alpha=0.4)

# ---------------------------------------------------------
# PLOT 4 (Empty)
# ---------------------------------------------------------
ax4.text(0.5, 0.5, "Empty for now", ha='center', va='center', fontsize=20, color='gray')
ax4.axis('off')


# Slider and Play Button
ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Segment (t)', 1, T_actual, valinit=T_actual, valstep=1, valfmt='%0.0f')

ax_play = plt.axes([0.85, 0.05, 0.08, 0.03])
btn_play = Button(ax_play, 'Play')

anim_running = False
anim = None

def animate(frame):
    current_val = int(slider.val)
    if current_val < T_actual:
        slider.set_val(current_val + 1)
    else:
        # Pause animation when we hit the end
        if anim_running:
            toggle_anim(None)

def toggle_anim(event):
    global anim_running, anim
    if anim_running:
        if anim is not None:
            anim.event_source.stop()
        anim_running = False
        btn_play.label.set_text('Play')
    else:
        # If at the end, restart from 1
        if int(slider.val) == T_actual:
            slider.set_val(1)
        
        if anim is None:
            anim = FuncAnimation(fig, animate, interval=500, cache_frame_data=False)
        else:
            anim.event_source.start()
        anim_running = True
        btn_play.label.set_text('Pause')
    fig.canvas.draw_idle()

btn_play.on_clicked(toggle_anim)

def update(val):
    t = int(slider.val)
    current_idx = t - 1
    
    # Update Plot 1
    line_true.set_data(steps[:t], true_remaining[:t])
    line_pred.set_data(steps[:t], predictions[:t])
    
    current_pred_total = t + predictions[current_idx]
    current_true_total = t + true_remaining[current_idx]
    vline_pred.set_xdata([current_pred_total, current_pred_total])
    vline_true.set_xdata([current_true_total, current_true_total])
    
    # Update Plot 2
    for i in range(T_actual):
        if i < t:
            traj_lines[i].set_visible(True)
            if i == t - 1:
                # Highlight the current segment
                traj_lines[i].set_color('red')
                traj_lines[i].set_linewidth(2.5)
                traj_lines[i].set_alpha(1.0)

            else:
                # Draw past segments fainter
                color_val = (i / max(1, t - 1)) * 0.85 
                traj_lines[i].set_color(cmap(color_val))
                traj_lines[i].set_linewidth(1.5)
                traj_lines[i].set_alpha(0.5)
        else:
            traj_lines[i].set_visible(False)
            
            
    # Update Plot 3
    s_W = np.atleast_1d(s_weights_cpu[current_idx])
    v_W = np.atleast_2d(v_weights_cpu[current_idx])
    
    bottom = np.zeros(T_actual)
    for i, container in enumerate(bar_containers):
        if t <= len(s_W) and t <= len(v_W):
            contribution = s_W * v_W[:, i]
        else:
            # Handle padding if s_W / v_W differ slightly in shape
            contribution = np.zeros(t)
            min_len = min(len(s_W), len(v_W), t)
            contribution[:min_len] = s_W[:min_len] * v_W[:min_len, i]
            
        for j, patch in enumerate(container.patches):
            if j < len(contribution):
                patch.set_height(contribution[j])
                patch.set_y(bottom[j]) # Apply stacked bottom
                bottom[j] += contribution[j]
            else:
                patch.set_height(0)
    
    fig.canvas.draw_idle()

# Initialize the plot with the max time
update(T_actual)
slider.on_changed(update)

plt.show()

