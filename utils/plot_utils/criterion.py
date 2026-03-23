import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_speed_comparison(df, file_name, tail=None):
    """ 
     Plots the original speed and the two computed speeds for a given file, along with their median, quartiles, mean, and standard deviation. 
     The statistics are displayed as horizontal lines on the plot and also summarized in a text box below each plot. 
     If 'tail' is specified, only the last 'tail' rows of the data will be plotted to focus on the end of life behavior.
     """
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig.suptitle(f"Speed Comparisons - {file_name}", fontsize=16)

    columns = ['Speed', 'ComputedSpeed_frames', 'ComputedSpeed_timestamp']
    colors = ['tab:blue', 'orange', 'green']
    titles = ['Original Speed', 'Computed Speed (Frames)', 'Computed Speed (Timestamp)']


    for i, col in enumerate(columns):
        if col in df.columns:
            data = df[col]
            median = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            mean = data.mean()
            std_dev = data.std()
            mean_plus_std = mean + std_dev

            plotted_data = data.tail(tail) if tail is not None else data


            # Plot data
            axes[i].plot(plotted_data, marker='o', linestyle='-', markersize=2, color=colors[i], alpha=0.6)
            
            # Add Horizontal Lines
            axes[i].axhline(median, color='red', linestyle='--', label=f'Median: {median:.2f}')
            axes[i].axhline(q1, color='black', linestyle=':', alpha=0.7, label=f'Q1: {q1:.2f}')
            axes[i].axhline(q3, color='black', linestyle=':', alpha=0.7, label=f'Q3: {q3:.2f}')
            axes[i].axhline(mean, color='blue', linestyle='-', alpha=0.7, label=f'Mean: {mean:.2f}')
            axes[i].axhline(mean_plus_std, color='purple', linestyle='-.', alpha=0.7, label=f'Mean + Std Dev: {mean_plus_std:.2f}')
            
            axes[i].set_title(titles[i])
            axes[i].set_xlabel("Row Number")
            axes[i].set_ylabel("Speed")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right', fontsize='small')

            # Add statistics text below the plot including std dev
            stats_text = fr"Median: {median:.2f} | IQR: {q3 - q1:.2f} | Mean: {mean:.2f} | Std Dev: $\mathbf{{{std_dev:.2f}}}$"
            axes[i].text(0.5, -0.25, stats_text, transform=axes[i].transAxes, 
                         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            axes[i].text(0.5, 0.5, f"'{col}' not found", ha='center', va='center')
            axes[i].axis('off')

    plt.show()



def compute_cumulative_distance(file_path, freq1=100, freq2=10, n=50):
    """ 
     Computes the cumulative distance from each point at index i (every freq1 rows) to the next n points at indices j (every freq2 rows). 
     The cumulative distances are plotted to visualize how they evolve over time.
        The cumulative distances are capped at the 3rd quartile to prevent outliers from dominating the plot.
        The function returns the list of cumulative distances for further analysis if needed.
        Parameters:
        - file_path: Path to the CSV file containing the data.
        - freq1: Frequency for selecting the starting points (default is 100).
        - freq2: Frequency for selecting the points to compare against (default is 10).
        - n: Number of points to compare against for each starting point (default is 50).

       """
    df = pd.read_csv(file_path)
    cumulative_distances = []
    for i in range(0, len(df) - freq1, freq1):
        # print(f"Processing row {i}...")
        p0 = df.loc[i, ['X', 'Y']].values
        cumulative_distance = 0
        for j in range(i + freq2, min(i + n*freq2, len(df)), freq2):
            # print(f"  Comparing to row {j}...")
            pi = df.loc[j, ['X', 'Y']].values
            distance = np.linalg.norm(pi - p0)
            cumulative_distance += distance
        cumulative_distances.append(cumulative_distance)
        # print(f"  Cumulative distance for row {i}: {cumulative_distance:.2f}")

    # Cap the cumulative distances to 3 quartiles to avoid outliers dominating the plot
    q3 = np.percentile(cumulative_distances, 75)
    cumulative_distances = [d if d <= q3 else q3 for d in cumulative_distances]

    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_distances, marker='o', linestyle='-', markersize=4)
    plt.title(f"Cumulative Distance - {os.path.basename(file_path)}")
    plt.xlabel(f"Index (every {freq1} rows)")
    plt.ylabel(f"Cumulative Distance on next {n} points every {freq2} rows")
    plt.grid(True, alpha=0.3)
    plt.show()
    return cumulative_distances

def compute_total_cumulative_distance(file_path, freq1=500, freq2=100):
    df = pd.read_csv(file_path)
    cumulative_distances = []
    for i in range(0, len(df) - freq1, freq1):
        # print(f"Processing row {i}...")
        p0 = df.loc[i, ['X', 'Y']].values
        cumulative_distance = 0
        for j in range(i + freq2, len(df), freq2):
            # print(f"  Comparing to row {j}...")
            pi = df.loc[j, ['X', 'Y']].values
            distance = np.linalg.norm(pi - p0)
            cumulative_distance += distance
        cumulative_distances.append(cumulative_distance)
        # print(f"  Cumulative distance for row {i}: {cumulative_distance:.2f}")
    
    # Cap the cumulative distances to 3 quartiles to avoid outliers dominating the plot
    q3 = np.percentile(cumulative_distances, 75)
    cumulative_distances = [d if d <= q3 else q3 for d in cumulative_distances]

    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_distances, marker='o', linestyle='-', markersize=4)
    plt.title(f"Cumulative Distance - {os.path.basename(file_path)}")
    plt.xlabel(f"Index (every {freq1} rows)")
    plt.ylabel(f"Cumulative Distance until end of dataset every {freq2} rows")
    plt.grid(True, alpha=0.3)
    plt.show()
    return cumulative_distances

def compute_total_smooth_cumulative_distance(file_path, freq1=500, freq2=100, smoothing_window=15):
    df = pd.read_csv(file_path)

    cumulative_distances_1 = []
    cumulative_distances_2 = []
    cumulative_distances_3 = []

    for i in range(0, len(df) - freq1, freq1):
        # print(f"Processing row {i}...")
        p0_1 = df.loc[i, ['X', 'Y']].values
        p0_2 = df.loc[i+smoothing_window, ['X', 'Y']].values
        p0_3 = df.loc[i+2*smoothing_window, ['X', 'Y']].values

        cumulative_distance_1 = 0
        cumulative_distance_2 = 0
        cumulative_distance_3 = 0

        for j in range(i + freq2, len(df)-2*smoothing_window, freq2):
            # print(f"  Comparing to row {j}...")
            pi_1 = df.loc[j, ['X', 'Y']].values
            pi_2 = df.loc[j+smoothing_window, ['X', 'Y']].values
            pi_3 = df.loc[j+2*smoothing_window, ['X', 'Y']].values

            distance_1 = np.linalg.norm(pi_1 - p0_1)
            distance_2 = np.linalg.norm(pi_2 - p0_2)
            distance_3 = np.linalg.norm(pi_3 - p0_3)

            cumulative_distance_1 += distance_1
            cumulative_distance_2 += distance_2
            cumulative_distance_3 += distance_3

        cumulative_distances_1.append(cumulative_distance_1)
        cumulative_distances_2.append(cumulative_distance_2)
        cumulative_distances_3.append(cumulative_distance_3)
        # print(f"  Cumulative distance for row {i}: {cumulative_distance:.2f}")

    cumulative_distances = [(d1 + d2 + d3) / 3 for d1, d2, d3 in zip(cumulative_distances_1, cumulative_distances_2, cumulative_distances_3)]
    
    # Cap the cumulative distances to 3 quartiles to avoid outliers dominating the plot
    q3 = np.percentile(cumulative_distances, 75)
    cumulative_distances = [d if d <= q3 else q3 for d in cumulative_distances]

    # add a different color for points below 2000, 1500, 1000, 750, 500, 250
    colors = []
    for d in cumulative_distances:
        if d < 250:
            colors.append('red')
        elif d < 500:
            colors.append('orange')
        elif d < 750:
            colors.append('yellow')
        elif d < 1000:
            colors.append('green')
        elif d < 1500:
            colors.append('cyan')
        elif d < 2000:
            colors.append('blue')
        else:
            colors.append('purple')


    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_distances, linestyle='-', color='black', alpha=0.3)
    plt.scatter(range(len(cumulative_distances)), cumulative_distances, c=colors, s=16, zorder=5)
    plt.title(f"Cumulative Distance - {os.path.basename(file_path)}")
    plt.xlabel(f"Index (every {freq1} rows)")
    plt.ylabel(f"Cumulative Distance until end of dataset every {freq2} rows")
    plt.grid(True, alpha=0.3)
    plt.show()
    return cumulative_distances