import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import random
import os

# --- Data Loading and Processing ---

def _fix_filepath_row(row):
    filename = row['Filename']
    terb = row['Terbinafine']
    m = re.match(r'^/(\d{8})_piworm(\d+)_(\d+)$', filename)
    if m:
        date, worm, idx = m.groups()
    else:
        m2 = re.search(r'coordinates_highestspeed_(\d{8})_(\d+)_(\d+)_with_time_speed', filename)
        if m2:
            date, worm, idx = m2.groups()
        else:
            return filename
    prefix = "TERBINAFINE+" if terb else "TERBINAFINE- (control)"

    if worm == "09" and date != "20240924": worm = "9" # special case
    
    return f"./data/{prefix}/coordinates_highestspeed_{date}_{worm}_{idx}_with_time_speed.csv"

def get_worm_trajectory(row):
    try:
        filepath = row["filepath"]
        if not os.path.exists(filepath):
             # Try absolute path if relative fails, or check if running from different dir
             if not os.path.exists(filepath):
                 print(f"File not found: {filepath}")
                 return None
        
        df = pd.read_csv(filepath, index_col="GlobalFrame")
        df = df.iloc[2:] # Skip first 2 rows
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp_s'] = df['Timestamp'].astype(np.int64) // 10**9
        df['relative_time'] = df['Timestamp_s'] - df['Timestamp_s'].min() # Relative time from start, in seconds
        return df
    except Exception as e:
        print(f"Error loading {row['filepath']}: {e}")
        return None

# --- Plotting Functions ---

def plot_traj(ax, df, title):
    if df is None: return
    sc = ax.scatter(df['X'], df['Y'], c=df['relative_time'], cmap='plasma', s=5, alpha=0.6)
    ax.plot(df['X'], df['Y'], "--", color='lightgray', linewidth=0.7, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal') # Important for trajectories
    return sc

def plot_time_series(ax, df, title, frame_of_death):
    if df is None: return
    ax.plot(df.index, df['X'], label='X Position')
    ax.plot(df.index, df['Y'], label='Y Position')
    ax.plot(df.index, df['Speed'], label='Speed', alpha=0.7)
    ax.vlines(frame_of_death, -100, 1000, color='g', linestyle='--', label='Frame of Death')
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position")
    ax.legend()

def plot_speed_around_death(ax, df1, worm1, df2, worm2, window=20):
    if df1 is None or df2 is None: return
    
    fc1 = int(worm1["LifespanInFrames"])
    fc2 = int(worm2["LifespanInFrames"])

    s1 = df1["Speed"].loc[fc1 - window: fc1 + window]
    s2 = df2["Speed"].loc[fc2 - window: fc2 + window]

    # convert index to frames relative to death
    x1 = s1.index - fc1
    x2 = s2.index - fc2

    if not s1.empty:
        ax.plot(x1, s1, marker='o', label=f'Worm 1 (ID: {worm1.name}, frame={fc1})', color='C0')
    
    if not s2.empty:
        ax.plot(x2, s2, marker='o', label=f'Worm 2 (ID: {worm2.name}, frame={fc2})', color='C1')

    ax.axvline(0, color='k', linestyle='--', label='Death frame')
    ax.set_xlabel('Frame relative to death (0 = death)')
    ax.set_ylabel('Speed')
    ax.set_title(f'Speed around death (±{window} frames)')
    ax.legend()
    ax.grid(alpha=0.3)

# --- Main Execution ---

def main():
    # Load summary
    try:
        summary = pd.read_csv('data/lifespan_summary.csv')
    except FileNotFoundError:
        print("Error: data/lifespan_summary.csv not found.")
        return

    summary['filepath'] = summary.apply(_fix_filepath_row, axis=1)

    # Select two random worms (one from each group for better comparison, or just random)
    # The prompt asks for "two random worms". To make the comparison interesting (like in the notebook),
    # I'll pick one Control and one Terbinafine, but if that's not possible, just two randoms.
    
    control_worms = summary[summary["Terbinafine"] == False]
    terb_worms = summary[summary["Terbinafine"] == True]
    
    if not control_worms.empty and not terb_worms.empty:
        worm1 = control_worms.sample(1).iloc[0]
        worm2 = terb_worms.sample(1).iloc[0]
        label1 = "Control"
        label2 = "Terbinafine"
    else:
        # Fallback if groups aren't available
        worms = summary.sample(2)
        worm1 = worms.iloc[0]
        worm2 = worms.iloc[1]
        label1 = "Worm 1"
        label2 = "Worm 2"

    print(f"Selected Worm 1: ID {worm1.name} ({label1})")
    print(f"Selected Worm 2: ID {worm2.name} ({label2})")

    df1 = get_worm_trajectory(worm1)
    df2 = get_worm_trajectory(worm2)

    if df1 is None or df2 is None:
        print("Failed to load trajectory data for one or both worms.")
        return

    # Create Plot
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.8])

    # Title with IDs
    fig.suptitle(f"Comparison of Random Worms\nWorm 1 ID: {worm1.name} ({label1}) | Worm 2 ID: {worm2.name} ({label2})", fontsize=16)

    # 1. Trajectories
    ax_traj1 = fig.add_subplot(gs[0, 0])
    sc1 = plot_traj(ax_traj1, df1, f"{label1} Trajectory (Lifespan: {worm1['LifespanInHours']:.1f}h)")
    plt.colorbar(sc1, ax=ax_traj1, label='Time (s)')

    ax_traj2 = fig.add_subplot(gs[0, 1])
    sc2 = plot_traj(ax_traj2, df2, f"{label2} Trajectory (Lifespan: {worm2['LifespanInHours']:.1f}h)")
    plt.colorbar(sc2, ax=ax_traj2, label='Time (s)')

    # 2. Time Series
    ax_ts1 = fig.add_subplot(gs[1, :])
    plot_time_series(ax_ts1, df1, f"{label1} Time Series (Lifespan: {worm1['LifespanInHours']:.1f}h)\n{'⚠️ Dried plate' if worm1['PlateHasDried'] else 'Healthy plate'}",
                     worm1["LifespanInFrames"])

    ax_ts2 = fig.add_subplot(gs[2, :])
    plot_time_series(ax_ts2, df2, f"{label2} Time Series (Lifespan: {worm2['LifespanInHours']:.1f}h)\n{'⚠️ Dried plate' if worm2['PlateHasDried'] else 'Healthy plate'}",
                     worm2["LifespanInFrames"])

    # 3. Speed around death
    ax_speed = fig.add_subplot(gs[3, :])
    plot_speed_around_death(ax_speed, df1, worm1, df2, worm2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    output_filename = "worm_comparison.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    main()
