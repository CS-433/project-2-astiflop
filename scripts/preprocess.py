import numpy as np
import pandas as pd
import os
import argparse
import cv2
from glob import glob

import tqdm

def check_first_rows(file_name, destination_dir):
    """
    Check if the 10 first rows of the CSV file have valid timestamps. If not, drop the ill ones and save the cleaned file back.
    Args:
        file (str): Filename to process
        destination_dir (str): Directory where the file is located
    """
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        rows_to_drop = []
        for i in range(min(10, len(df) - 1)):
            if pd.isna(df.loc[i, "Timestamp"]):
                rows_to_drop.append(i)
            elif pd.notna(df.loc[i + 1, "Timestamp"]):
                diff = abs((df.loc[i + 1, "Timestamp"] - df.loc[i, "Timestamp"]).total_seconds())
                if diff > 1000:
                    rows_to_drop.append(i)
                    
        if rows_to_drop:
            df = df.drop(index=rows_to_drop).reset_index(drop=True)
        
        file_output_path = os.path.join(destination_dir, os.path.basename(file_name))
        df.to_csv(file_output_path, index=False)
        return file_output_path


def add_segment_column(file, frames_per_segment=900):
    """
    Add a 'Segment' column to each CSV file, indicating the segment number based on frame count.

    Args:
        file (str): Filename to process
        frame_per_segment (int): Number of frames per segment
    """
    df = pd.read_csv(file)
    df["Segment"] = df["GlobalFrame"] // frames_per_segment
    df.to_csv(file, index=False)


def add_turning_rate_column_within_segments(
    df: pd.DataFrame, speed_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Calculates the 'Turning_rate' (Delta Theta) for each time step within the
    trajectories and applies a speed filter to remove angular noise when the worm is stopped.

    Args:
        df (pd.DataFrame): DataFrame containing worm tracking data with 'X', 'Y', 'Speed', and 'Segment' columns.
        speed_threshold (float): Minimum speed required for an angular change to be considered valid (e.g., 0.05).

    Returns:
        pd.DataFrame: DataFrame with the 'Turning_rate' column added.
    """
    df = df.copy()

    # --- 1. Calculate absolute heading (Theta) ---
    # Calculate differences in x and y coordinates (Displacement vectors)
    df["dx"] = df["X"].diff()
    df["dy"] = df["Y"].diff()

    # Calculate the absolute angle (heading) using atan2 to handle all quadrants
    df["theta"] = np.arctan2(df["dy"], df["dx"])

    # --- 2. Calculate the change in angle (Delta Theta) ---
    # Delta theta is the difference between consecutive headings
    df["delta_theta"] = df["theta"].diff()

    # 3. Adjust delta_theta to be within [-pi, pi] (Handling the circular nature of angles)
    # This prevents turns from 359 to 1 degree from being calculated as -358 degrees.
    df["delta_theta"] = (df["delta_theta"] + np.pi) % (2 * np.pi) - np.pi

    # --- 4. Apply Speed Threshold Mask (Noise Cleaning) ---

    # Identify time steps where the speed is below the established noise floor
    mask_stopped = df["Speed"] <= speed_threshold

    # Set the angular change to 0.0 for those stopped instances
    # This removes spurious delta_theta values caused by sensor/camera noise when the worm is immobile.
    df.loc[mask_stopped, "delta_theta"] = 0.0

    # 5. Handle initial NaNs
    # The first 2 rows of any segment will have NaNs due to the two consecutive diff() operations.
    # We replace them with 0.0, assuming no significant turn at the very start of the tracking.
    df["Turning_rate"] = df["delta_theta"].fillna(0.0)

    # --- 6. Cleanup and Return ---
    # Drop intermediate columns for a cleaner output
    df = df.drop(columns=["dx", "dy", "theta", "delta_theta"])

    return df


def cap_speed(df, speed_cap=10):
    """
    Cap the 'Speed' values in the DataFrame to a maximum value.

    Args:
        df (pd.DataFrame)
        speed_cap (float): Maximum speed value
    """
    df["Speed"] = df["Speed"].clip(upper=speed_cap)
    return df


def remove_high_speed_outliers(df, speed_threshold=4):
    """
    Remove rows with speed above the threshold and adjust subsequent coordinates
    to stitch the trajectory, effectively removing the jump.

    Args:
        df (pd.DataFrame): DataFrame containing worm tracking data.
        speed_threshold (float): Speed threshold for outlier detection.

    Returns:
        pd.DataFrame: DataFrame with outliers removed and coordinates adjusted.
    """
    df = df.copy()

    # Identify outliers
    outlier_mask = (df["Speed"] > speed_threshold) | df["Speed"].isna()

    if not outlier_mask.any():
        return df

    # Calculate displacements (jumps)
    dx = df["X"].diff().fillna(0)
    dy = df["Y"].diff().fillna(0)

    # Identify shifts caused by outliers
    shifts_x = dx.where(outlier_mask, 0)
    shifts_y = dy.where(outlier_mask, 0)

    # Calculate cumulative shift
    cum_shift_x = shifts_x.cumsum()
    cum_shift_y = shifts_y.cumsum()

    # Apply adjustment
    df["X"] -= cum_shift_x
    df["Y"] -= cum_shift_y

    # Drop the outlier rows
    df = df[~outlier_mask].reset_index(drop=True)

    return df


def remove_large_displacement_outliers(df, distance_threshold=2):
    """
    Remove rows where the displacement (distance) from the previous frame exceeds the threshold,
    and adjust subsequent coordinates to stitch the trajectory.

    Args:
        df (pd.DataFrame): DataFrame containing worm tracking data.
        distance_threshold (float): Distance threshold for outlier detection.

    Returns:
        pd.DataFrame: DataFrame with outliers removed and coordinates adjusted.
    """
    df = df.copy()

    # Calculate displacements (jumps)
    dx = df["X"].diff().fillna(0)
    dy = df["Y"].diff().fillna(0)

    # Calculate Euclidean distance
    distances = np.sqrt(dx**2 + dy**2)

    # Identify outliers
    outlier_mask = distances > distance_threshold

    if not outlier_mask.any():
        return df

    # Identify shifts caused by outliers
    shifts_x = dx.where(outlier_mask, 0)
    shifts_y = dy.where(outlier_mask, 0)

    # Calculate cumulative shift
    cum_shift_x = shifts_x.cumsum()
    cum_shift_y = shifts_y.cumsum()

    # Apply adjustment
    df["X"] -= cum_shift_x
    df["Y"] -= cum_shift_y

    # Drop the outlier rows
    df = df[~outlier_mask].reset_index(drop=True)

    return df


def filter_and_transform_to_displacement(df, distance_threshold=2):
    """
    1. Drop rows with NaN in X or Y.
    2. Replace X and Y with their displacement from the previous row.
    3. Add a 'Displacement' column with the Euclidean distance.
    4. Drop rows where 'Displacement' exceeds the threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        distance_threshold (float): Displacement threshold.

    Returns:
        pd.DataFrame: Transformed and filtered dataframe.
    """
    # 1. Drop NA for X or Y
    df = df.dropna(subset=["X", "Y"]).copy()

    # Calculate displacements
    dx = df["X"].diff()
    dy = df["Y"].diff()

    # 3. Displacement column
    df["Displacement"] = np.sqrt(dx**2 + dy**2)

    # 2. Puts the displacement on X, Y
    df["X"] = dx
    df["Y"] = dy

    # 4. Drop rows with displacement above threshold
    # We also drop the first row because diff() produces NaN
    df = df.dropna(subset=["Displacement"])
    df = df[df["Displacement"] <= distance_threshold]

    return df


def filter_and_reconstruct_coordinates(df, distance_threshold=2):
    """
    1. Drop rows with NaN in X or Y.
    2. Calculate displacement from the previous row.
    3. Drop rows where displacement exceeds the threshold.
    4. Reconstruct X and Y by cumulatively summing the valid displacements.

    Args:
        df (pd.DataFrame): Input dataframe.
        distance_threshold (float): Displacement threshold.

    Returns:
        pd.DataFrame: Transformed and filtered dataframe.
    """
    # 1. Drop NA for X or Y
    df = df.dropna(subset=["X", "Y"]).copy()

    if df.empty:
        return df

    # Store start coordinates
    start_x = df.iloc[0]["X"]
    start_y = df.iloc[0]["Y"]

    # Calculate displacements
    # fillna(0) ensures the first row (displacement 0) is kept
    df["dx"] = df["X"].diff().fillna(0)
    df["dy"] = df["Y"].diff().fillna(0)

    # 3. Displacement column
    df["Displacement"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    # 4. Drop rows with displacement above threshold
    df = df[df["Displacement"] <= distance_threshold]

    # 5. Reconstruct coordinates
    df["X"] = df["dx"].cumsum() + start_x
    df["Y"] = df["dy"].cumsum() + start_y

    # Drop intermediate columns
    df = df.drop(columns=["dx", "dy", "Displacement"])

    return df


def filter_and_reconstruct_coordinates_by_segment(df, distance_threshold=2):
    """
    Applies filter_and_reconstruct_coordinates to each segment individually.

    Args:
        df (pd.DataFrame): Input dataframe.
        distance_threshold (float): Displacement threshold.

    Returns:
        pd.DataFrame: Transformed and filtered dataframe.
    """
    cleaned_segments = []
    for _, segment_df in df.groupby("Segment"):
        cleaned_segments.append(
            filter_and_reconstruct_coordinates(segment_df, distance_threshold)
        )

    if not cleaned_segments:
        return df

    return pd.concat(cleaned_segments).reset_index(drop=True)

def clean_segment_gaps(
    segment_df, gap_interpolation_limit=10, long_gap_threshold=11, drop_na=False
):
    """Repair short gaps by interpolation and assume long gaps indicate the worm is not moving, so we remplace them with 0 speed and
    We drop them if drop_na is True.

    Args:
        segment_df: DataFrame of a single segment containing columns `x`, `y`,
            and `speed` where gaps may be NaN.

    Returns:
        pandas.DataFrame: Cleaned segment dataframe with short gaps interpolated
        and rows from long gaps removed.
    """
    gap_mask = segment_df[["X", "Y", "Speed"]].isna().any(axis=1)
    is_nan = gap_mask.astype(int)
    starts = (is_nan.diff() == 1).astype(int)
    if len(is_nan) > 0 and is_nan.iloc[0] == 1:
        starts.iloc[0] = 1

    gap_ids = starts.cumsum() * is_nan

    rows_to_remove = []
    for gap_id in gap_ids[gap_ids > 0].unique():
        indices = segment_df.index[gap_ids == gap_id].tolist()
        gap_size = len(indices)

        if gap_size <= gap_interpolation_limit:
            if len(indices) > 0:
                start_idx = max(segment_df.index.min(), indices[0] - 1)
                end_idx = min(segment_df.index.max(), indices[-1] + 1)
                segment_df.loc[start_idx:end_idx, ["X", "Y", "Speed"]] = segment_df.loc[
                    start_idx:end_idx, ["X", "Y", "Speed"]
                ].interpolate(method="linear", limit_direction="both")
        elif gap_size >= long_gap_threshold:
            rows_to_remove.extend(indices)

    if rows_to_remove:
        segment_df = segment_df.drop(index=rows_to_remove).reset_index(drop=True)

    return segment_df


def normalize_coordinates(df):
    """
    Normalize coordinates to [0,1] range.
    """
    df["X"] = (df["X"] - df["X"].min()) / (df["X"].max() - df["X"].min())
    df["Y"] = (df["Y"] - df["Y"].min()) / (df["Y"].max() - df["Y"].min())
    return df


def normalize_coordinates_and_speed(df):
    """
    Applies Min-Max normalization (scaling to a 0-1 range) to the fixed
    columns: 'X', 'Y', 'Speed', 'ComputedSpeed_frames', and 'ComputedSpeed_timestamp'.
    """

    columns_to_normalize = [
        "X",
        "Y",
        "Speed",
        "ComputedSpeed_frames",
        "ComputedSpeed_timestamp",
    ]

    df_normalized = df.copy()

    for column in columns_to_normalize:
        # Check if the column exists
        if column not in df_normalized.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue

        # Calculate min/max and apply the formula
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        denominator = max_val - min_val

        # Avoid division by zero
        if denominator == 0:
            print(
                f"Note: Column '{column}' has a constant value. Setting normalized value to 0."
            )
            df_normalized[column] = 0.0
        else:
            # Min-Max normalization formula: (X - min) / (max - min)
            df_normalized[column] = (df_normalized[column] - min_val) / denominator

    return df_normalized


def add_computed_speed_columns(df):
    """
    Calculate new speeds based on recomputed X and Y.
    Adds 'ComputedSpeed_frames' and 'ComputedSpeed_timestamp'.
    """
    df = df.copy()

    # Calculate displacement
    dx = df["X"].diff()
    dy = df["Y"].diff()
    displacement = np.sqrt(dx**2 + dy**2)

    # Calculate time diff for frames
    dt_frames = df["LocalFrame"].diff()

    # Calculate time diff for timestamp
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    dt_timestamp = df["Timestamp"].diff().dt.total_seconds()

    # Calculate speeds
    df["ComputedSpeed_frames"] = displacement / dt_frames
    df["ComputedSpeed_timestamp"] = displacement / dt_timestamp

    # Fill NaNs (first row of segment) with 0.0
    df["ComputedSpeed_frames"] = df["ComputedSpeed_frames"].fillna(0.0)
    df["ComputedSpeed_timestamp"] = df["ComputedSpeed_timestamp"].fillna(0.0)

    return df


def preprocess_file(
    file, speed_cap=4, normalize_coords=False, distance_threshold=16
):
    """
    Preprocess a single CSV file by applying various cleaning steps.

    Args:
        file (str): Filename to process
        speed_cap (float): Maximum speed value
        normalize_coords (bool): Whether to normalize coordinates
        distance_threshold (float): Distance threshold for coordinate reconstruction
    """
    df = pd.read_csv(file)
    df = filter_and_reconstruct_coordinates_by_segment(
        df, distance_threshold=distance_threshold
    )
    cleaned_segments = []
    for segment_id, segment_df in df.groupby("Segment"):
        segment_df = add_computed_speed_columns(segment_df)
        segment_df = add_turning_rate_column_within_segments(segment_df)
        cleaned_segments.append(segment_df)

    cleaned_df = pd.concat(cleaned_segments).reset_index(drop=True)

    if normalize_coords:
        # cleaned_df = normalize_coordinates(cleaned_df)
        cleaned_df = normalize_coordinates_and_speed(cleaned_df)

    cleaned_df.to_csv(file, index=False)


def process_all_files(
    data,
    output_dir="preprocessed_data/",
    speed_cap=4,
    normalize_coords=False,
    specific_file=None,
    distance_threshold=16,
    input_base_dir="data",
):
    """
    Preprocess all CSV files in the specified treatment group.

    Args:
        data (str): Data directory name
        output_dir (str): Directory to save preprocessed files
        speed_cap (float): Maximum speed value
        normalize_coords (bool): Whether to normalize coordinates
        specific_file (str): Optional specific file to process (basename)
        distance_threshold (float): Distance threshold for coordinate reconstruction
        input_base_dir (str): Directory containing the data folders
    """
    destination_dir = os.path.join(output_dir, data)
    os.makedirs(destination_dir, exist_ok=True)

    if specific_file:
        # Check if the file exists in this treatment folder
        full_path = os.path.join(input_base_dir, data, specific_file)
        if os.path.exists(full_path):
            files = [full_path]
        else:
            return None
    else:
        files = glob(os.path.join(input_base_dir, data, "*.csv"))

    pbar = tqdm.tqdm(files)
    for file in pbar:
        pbar.set_description(f"Processing {os.path.basename(file)}")
        output_path = check_first_rows(file, destination_dir)
        add_segment_column(output_path)
        preprocess_file(
            output_path,
            speed_cap=speed_cap,
            normalize_coords=normalize_coords,
            distance_threshold=distance_threshold,
        )


def get_global_stats(input_dir, speed_column="ComputedSpeed_frames", window_size=150):
    """
    Iterates through ALL CSV files in input_dir to find:
    1. Global Min/Max Speed.
    2. 'Max Span': The maximum spatial area covered by a worm within a window of 'window_size',
       calculated PER SEGMENT to avoid jumps between segments.
    """
    all_files = glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)

    min_speed, max_speed = float("inf"), float("-inf")
    max_span = 0.0

    print(f"Calculating global statistics on {len(all_files)} files...")

    for file in tqdm.tqdm(all_files, desc="Global Stats"):
        try:
            df = pd.read_csv(file)
            if df.empty or speed_column not in df.columns:
                continue

            # 1. Speed Stats (Safe to calculate on the whole dataframe)
            min_speed = min(min_speed, df[speed_column].min())
            max_speed = max(max_speed, df[speed_column].max())

            # 2. Calculate 'Max Span' PER SEGMENT
            # We must group by segment, otherwise the rolling window will calculate
            # the distance between the end of Segment N and start of Segment N+1,
            # which could be huge and incorrect.
            for _, seg_df in df.groupby("Segment"):
                if len(seg_df) < window_size:
                    continue

                # Calculate rolling min and max for X within the segment
                rolling_max_x = seg_df["X"].rolling(window=window_size).max()
                rolling_min_x = seg_df["X"].rolling(window=window_size).min()
                span_x = (rolling_max_x - rolling_min_x).max()

                # Calculate rolling min and max for Y within the segment
                rolling_max_y = seg_df["Y"].rolling(window=window_size).max()
                rolling_min_y = seg_df["Y"].rolling(window=window_size).min()
                span_y = (rolling_max_y - rolling_min_y).max()

                # Update global max_span if valid
                local_max_span = max(span_x, span_y)
                if not np.isnan(local_max_span):
                    max_span = max(max_span, local_max_span)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Add a 10% safety margin
    max_span = max_span * 1.1

    return {"v_min": min_speed, "v_max": max_speed, "max_span": max_span}


def create_multichannel_image(
    clip_df,
    global_stats,
    output_path,
    img_size=128,
    speed_column="ComputedSpeed_frames",
):
    """
    Creates a 3-channel tensor image using OpenCV with CENTERED ZOOM.

    Channel 0 (Blue): Speed (Normalized globally)
    Channel 1 (Green): Time (Normalized locally 0->255)
    Channel 2 (Red): Path/Occupancy (Binary)
    """
    # 1. Initialize black image
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 2. Extract Data
    x = clip_df["X"].values
    y = clip_df["Y"].values
    speed = clip_df[speed_column].values
    num_points = len(x)

    # --- CENTERING LOGIC ---
    # Find the geometric center of THIS specific clip
    center_x = (np.max(x) + np.min(x)) / 2
    center_y = (np.max(y) + np.min(y)) / 2

    # Retrieve the global max span (the zoom factor)
    span = global_stats["max_span"]
    if span == 0:
        span = 1  # Avoid division by zero

    # Mapping:
    # 1. Center the coordinates around 0: (val - center)
    # 2. Normalize by the global max span: / span
    # 3. Scale to image size: * img_size
    # 4. Shift to image center: + img_size / 2
    x_pix = ((x - center_x) / span * img_size + img_size / 2).astype(int)
    y_pix = ((y - center_y) / span * img_size + img_size / 2).astype(int)

    # Clip coordinates to ensure they stay within image bounds
    x_pix = np.clip(x_pix, 0, img_size - 1)
    y_pix = np.clip(y_pix, 0, img_size - 1)

    # 4. Normalize Speed (Global scale is kept for speed consistency)
    speed_norm = (speed - global_stats["v_min"]) / (
        global_stats["v_max"] - global_stats["v_min"]
    )
    speed_norm = np.clip(speed_norm, 0.0, 1.0) * 255

    # 5. Draw Trajectory
    for i in range(num_points - 1):
        pt1 = (x_pix[i], y_pix[i])
        pt2 = (x_pix[i + 1], y_pix[i + 1])

        # Skip if points are identical (no movement)
        if pt1 == pt2:
            continue

        # Channel 0 (Blue): SPEED (Average of two points)
        val_speed = int((speed_norm[i] + speed_norm[i + 1]) / 2)

        # Channel 1 (Green): TIME (Gradient 0->255)
        val_time = int((i / num_points) * 255)

        # Channel 2 (Red): PATH (Always 255)
        val_path = 255

        color = (val_speed, val_time, val_path)
        cv2.line(image, pt1, pt2, color, thickness=2)

    # 6. Save
    cv2.imwrite(output_path, image)


def preprocess_for_cnn(
    input_dir="preprocessed_data",
    output_dir="cnn_dataset",
    window_size=150,
    stride=75,
    img_size=128,
    speed_column="ComputedSpeed_frames",
):
    """
    Main function for CNN dataset generation.
    Iterates through each segment separately and uses 'window_size' to calculate global stats.
    """

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"Error: Directory {input_dir} is empty. Cannot generate images.")
        return

    print("--- Step 1/2: Calculating global scales (Speed & Spatial Span) ---")
    # PASS window_size to get_global_stats
    global_stats = get_global_stats(input_dir, speed_column, window_size=window_size)
    print("Global Stats found:", global_stats)

    print(f"--- Step 2/2: Generating Multichannel Images ---")

    files = glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    count_generated = 0

    for file_path in tqdm.tqdm(files, desc="Processing Files"):
        try:
            df = pd.read_csv(file_path)

            treatment_name = (
                "TERBINAFINE+"
                if df["Terbinafine"].iloc[0]
                else "TERBINAFINE- (control)"
            )
            worm_id = df["WormID"].iloc[0]

            save_dir = os.path.join(
                output_dir, treatment_name, worm_id, "photos_trajectories"
            )
            os.makedirs(save_dir, exist_ok=True)

            # Group by Segment to handle discontinuities
            for segment_id, segment_df in df.groupby("Segment"):

                segment_df = segment_df.reset_index(drop=True)
                n_rows_seg = len(segment_df)

                # Sliding window INSIDE the segment
                for start_idx in range(0, n_rows_seg - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    clip = segment_df.iloc[start_idx:end_idx].copy()

                    # Check for NaNs
                    cols_to_check = ["X", "Y", speed_column]
                    if clip[cols_to_check].isnull().values.any():
                        continue

                    # Naming includes Segment ID
                    img_name = f"seg_{segment_id}_frame_{start_idx}_to_{end_idx}.png"
                    img_path = os.path.join(save_dir, img_name)

                    create_multichannel_image(
                        clip, global_stats, img_path, img_size, speed_column
                    )
                    count_generated += 1

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"Done! {count_generated} images generated.")



def load_config(data_dir):
    config_path = os.path.join(data_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {data_dir}. A config.json file is required with 'control_folder', 'treated_folder', and 'lifespan_file' keys.")
    
    with open(config_path, "r") as f:
        config = pd.read_json(config_path, typ='series')
    return config

def calculate_and_save_stats(output_dir, speed_column="ComputedSpeed_frames"):
    """
    Calculate global statistics (mean, std, min, max) for feature columns 
    across all preprocessed CSV files and save to a JSON file.
    """
    all_files = glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)
    
    # Initialize accumulators
    total_count = 0
    sum_vals = None
    sum_sq_vals = None
    min_vals = None
    max_vals = None
    
    feature_cols = ["X", "Y", speed_column]
    
    print("Calculating dataset statistics...")
    
    for file_path in tqdm.tqdm(all_files, desc="Stats"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Filter for numeric columns of interest
            cols = [c for c in feature_cols if c in df.columns]
            if not cols:
                continue
                
            data = df[cols].values
            
            if sum_vals is None:
                final_cols = cols
                sum_vals = np.zeros(len(cols))
                sum_sq_vals = np.zeros(len(cols))
                min_vals = np.full(len(cols), np.inf)
                max_vals = np.full(len(cols), -np.inf)
                
            if cols != final_cols:
                # Skip files with different columns if mismatch
                continue

            # Update stats
            sum_vals += np.sum(data, axis=0)
            sum_sq_vals += np.sum(data ** 2, axis=0)
            min_vals = np.minimum(min_vals, np.min(data, axis=0))
            max_vals = np.maximum(max_vals, np.max(data, axis=0))
            total_count += len(data)

        except Exception as e:
            print(f"Error processing {file_path} for stats: {e}")

    if total_count > 0:
        mean = sum_vals / total_count
        std = np.sqrt(sum_sq_vals / total_count - mean ** 2)
        
        stats = {
            "features": final_cols,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_vals.tolist(),
            "max": max_vals.tolist(),
            "count": int(total_count)
        }
        
        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, "w") as f:
            pd.Series(stats).to_json(f)
        print(f"Saved dataset statistics to {stats_path}")
    else:
        print("No data found to calculate statistics.")


def parse_args():
    parser = argparse.ArgumentParser(description="Process files with options.")
    parser.add_argument("data_dir", type=str, help="Root directory containing data and config.json")
    parser.add_argument("--speed-cap", type=float, default=4, help="Max speed cap.")
    parser.add_argument(
        "--distance-threshold", type=float, default=16, help="Distance threshold."
    )
    parser.add_argument(
        "--output-dir", type=str, default="preprocessed_data/", help="Output for CSVs."
    )
    parser.add_argument(
        "--cnn-output-dir", type=str, default="cnn_dataset/", help="Output for Images."
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize coordinates in CSV."
    )
    parser.add_argument("--file", type=str, help="Specific file.")

    # Arguments for flow control
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate images for CNN after CSV processing.",
    )
    parser.add_argument(
        "--only-cnn",
        action="store_true",
        help="Skip CSV preprocessing and ONLY generate CNN images.",
    )

    parser.set_defaults(normalize=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.data_dir)
        CONTROL_DIR = config.get("control_folder", "TERBINAFINE- (control)")
        TREATED_DIR = config.get("treated_folder", "TERBINAFINE+")
    except Exception as e:
        print(f"Error loading config of data: {e}")
        exit(1)

    # 1. CSV Preprocessing Step
    # Run only if NOT in 'only-cnn' mode
    if not args.only_cnn:
        print(f"------ CSV Processing: Control Group ({CONTROL_DIR}) ------")
        process_all_files(
            CONTROL_DIR,
            output_dir=args.output_dir,
            speed_cap=args.speed_cap,
            normalize_coords=args.normalize,
            specific_file=args.file,
            distance_threshold=args.distance_threshold,
            input_base_dir=args.data_dir
        )

        print(f"------ CSV Processing: Treated Group ({TREATED_DIR}) ------")
        process_all_files(
            TREATED_DIR,
            output_dir=args.output_dir,
            speed_cap=args.speed_cap,
            normalize_coords=args.normalize,
            specific_file=args.file,
            distance_threshold=args.distance_threshold,
            input_base_dir=args.data_dir
        )
        
        # Calculate statistics after processing
        if not args.file:
            calculate_and_save_stats(args.output_dir)

    else:
        print("Skipping CSV Preprocessing (--only-cnn active)...")

    # 2. Image Generation Step
    # Run if explicitly requested OR if 'only-cnn' is active
    if args.generate_images or args.only_cnn:
        print("\n------ Starting image generation for CNN ------")
        preprocess_for_cnn(
            input_dir=args.output_dir,
            output_dir=args.cnn_output_dir,
            window_size=300,
            stride=150,
            img_size=128,
            speed_column="ComputedSpeed_frames",
        )

