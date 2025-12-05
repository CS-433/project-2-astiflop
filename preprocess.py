import numpy as np
import pandas as pd
import os
import argparse
from glob import glob

CONTROL = "TERBINAFINE- (control)"
TREATED = "TERBINAFINE+"

def rename_file(file_name, treatment, treatment_dir):
    """
    Rename files according to worm ID to ensure consistency with lifespan_summary.csv

    Args:
        file (str): Original filename
        treatment (str): Treatment group
        treatment_dir (str): Directory to save renamed files
    """
    if file_name.endswith(".csv"):
        parts = file_name.split("_")
        if len(str(parts[3])) == 1:
            parts[3] = "0" + str(parts[3])
        worm_id = f"{parts[2]}_piworm{parts[3]}_{parts[4]}"
        df = pd.read_csv(os.path.join("data", treatment, file_name))
        new_path = os.path.join(treatment_dir, f"{worm_id}.csv")
        df.to_csv(new_path, index=False)
        return new_path
    return None


def drop_first_row(file_name, treatment_dir):
    """
    Drop the first row of the CSV file, which may contain invalid data.
    Args:
        file (str): Filename to process
        treatment_dir (str): Directory where the file is located
    """
    if file_name.endswith(".csv"):
        file_path = os.path.join(treatment_dir, file_name)
        df = pd.read_csv(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        if (
            pd.isna(df.loc[0, "Timestamp"])
            or abs((df.loc[1, "Timestamp"] - df.loc[0, "Timestamp"]).total_seconds())
            > 3600
        ):
            df = df.drop(index=0).reset_index(drop=True)
            df.to_csv(file_path, index=False)


def add_segment_column(file, frames_per_segment=900):
    """
    Add a 'Segment' column to each CSV file, indicating the segment number based on frame count.

    Args:
        file (str): Filename to process
        frame_per_segment (int): Number of frames per segment
    """
    df = pd.read_csv(file)
    num_rows = len(df)
    df["Segment"] = np.arange(num_rows) // frames_per_segment
    df.to_csv(file, index=False)


def add_label_column(file, label):
    """
    Add a 'Label' column to the CSV file with the specified label.

    Args:
        file (str): Filename to process
        label (int): Label to add (0 for control, 1 for treated)
    """
    df = pd.read_csv(file)
    df["Terbinafine"] = label
    df.to_csv(file, index=False)


def add_worm_id_column(file, worm_id):
    """
    Add a 'WormID' column to the CSV file with the specified worm ID.

    Args:
        file (str): Filename to process
        worm_id (str): Worm ID to add
    """
    df = pd.read_csv(file)
    df["WormID"] = worm_id
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
    df["Displacement"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

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
        cleaned_segments.append(filter_and_reconstruct_coordinates(segment_df, distance_threshold))

    if not cleaned_segments:
        return df

    return pd.concat(cleaned_segments).reset_index(drop=True)


def drop_frames_after_death(df, frame_of_death):
    """
    Drop all frames after the frame of death.

    Args:
        df (pd.DataFrame)
        frame_of_death (int): Frame number indicating death
    """
    return df[df["GlobalFrame"] <= frame_of_death]


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

def preprocess_file(file, frame_of_death, speed_cap=4, normalize_coords=False, distance_threshold=10):
    """
    Preprocess a single CSV file by applying various cleaning steps.

    Args:
        file (str): Filename to process
        frame_of_death (int): Frame number indicating death
        speed_cap (float): Maximum speed value
        normalize_coords (bool): Whether to normalize coordinates
        distance_threshold (float): Distance threshold for coordinate reconstruction
    """
    df = pd.read_csv(file)
    df = drop_frames_after_death(df, frame_of_death)
    # df = cap_speed(df, speed_cap=speed_cap) # original preprocessing
    # df = remove_high_speed_outliers(df) # preprocessing 2 
    # df = remove_large_displacement_outliers(df, distance_threshold=2) # preprocessing 3
    # df = filter_and_transform_to_displacement(df, distance_threshold=2) # preprocessing 4
    # df = filter_and_reconstruct_coordinates(df, distance_threshold=2) # preprocessing 5
    # df = filter_and_reconstruct_coordinates_by_segment(df, distance_threshold=2) # preprocessing 6
    # df = filter_and_reconstruct_coordinates_by_segment(df, distance_threshold=20) # preprocessing 7
    df = filter_and_reconstruct_coordinates_by_segment(df, distance_threshold=distance_threshold) # final preprocessing
    cleaned_segments = []
    for segment_id, segment_df in df.groupby("Segment"):
        # segment_df = clean_segment_gaps(segment_df) # known useless for preprocessing 6+
        segment_df = add_turning_rate_column_within_segments(segment_df)
        cleaned_segments.append(segment_df)

    cleaned_df = pd.concat(cleaned_segments).reset_index(drop=True)

    if normalize_coords:
        cleaned_df = normalize_coordinates(cleaned_df)

    cleaned_df.to_csv(file, index=False)


def process_all_files(
    treatment, lifespan_summary, output_dir="preprocessed_data/", speed_cap=4, normalize_coords=False, specific_file=None, distance_threshold=10
):
    """
    Preprocess all CSV files in the specified treatment group.

    Args:
        treatment (str): Treatment group
        lifespan_summary (pd.DataFrame): DataFrame containing lifespan summary
        output_dir (str): Directory to save preprocessed files
        speed_cap (float): Maximum speed value
        normalize_coords (bool): Whether to normalize coordinates
        specific_file (str): Optional specific file to process (basename)
        distance_threshold (float): Distance threshold for coordinate reconstruction
    """
    treatment_dir = os.path.join(output_dir, treatment)
    os.makedirs(treatment_dir, exist_ok=True)

    if specific_file:
        # Check if the file exists in this treatment folder
        full_path = os.path.join("data", treatment, specific_file)
        if os.path.exists(full_path):
            files = [full_path]
        else:
            files = []
    else:
        files = glob(os.path.join("data", treatment, "*.csv"))

    preprocessed_files = []
    for file in files:
        new_path = rename_file(os.path.basename(file), treatment, treatment_dir)
        if new_path:
            preprocessed_files.append(new_path)

    for file in preprocessed_files:
        drop_first_row(os.path.basename(file), treatment_dir)
        add_segment_column(file)
        add_label_column(file, label=treatment == TREATED)
        worm_id = os.path.splitext(os.path.basename(file))[0]
        add_worm_id_column(file, worm_id)

    for file in preprocessed_files:
        print(f"Processing {os.path.basename(file)}...")
        worm_id = os.path.splitext(os.path.basename(file))[0]
        frame_of_death = lifespan_summary.loc[
            lifespan_summary["Filename"] == "/" + worm_id, "LifespanInFrames"
        ].values[0]
        preprocess_file(
            file,
            frame_of_death,
            speed_cap=speed_cap,
            normalize_coords=normalize_coords,
            distance_threshold=distance_threshold,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Process files with options.")
    parser.add_argument(
        "--speed-cap",
        type=float,
        default=4,
        help="Maximum speed value to cap at (default: 4).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=10,
        help="Distance threshold for coordinate reconstruction (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="preprocessed_data/",
        help="Directory to save preprocessed files (default: preprocessed_data/).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Do normalize coordinates.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do not normalize coordinates.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific file to process (basename).",
    )
    parser.set_defaults(normalize=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    lifespan_summary = pd.read_csv("data/lifespan_summary.csv")

    process_all_files(
        CONTROL,
        lifespan_summary,
        output_dir=args.output_dir,
        speed_cap=args.speed_cap,
        normalize_coords=args.normalize,
        specific_file=args.file,
        distance_threshold=args.distance_threshold,
    )

    process_all_files(
        TREATED,
        lifespan_summary,
        output_dir=args.output_dir,
        speed_cap=args.speed_cap,
        normalize_coords=args.normalize,
        specific_file=args.file,
        distance_threshold=args.distance_threshold,
    )
    print("Preprocessing completed.")
