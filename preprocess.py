import numpy as np
import pandas as pd
import os
import argparse
from glob import glob

CONTROL = "TERBINAFINE- (control)"
TREATED = "TERBINAFINE+"
PREPROCESSED_DIR = "preprocessed_data/"


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
        df.to_csv(os.path.join(treatment_dir, f"{worm_id}.csv"), index=False)


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


def cap_speed(df, speed_cap=10):
    """
    Cap the 'Speed' values in the DataFrame to a maximum value.

    Args:
        df (pd.DataFrame)
        speed_cap (float): Maximum speed value
    """
    df["Speed"] = df["Speed"].clip(upper=speed_cap)
    return df


def drop_frames_after_death(df, frame_of_death):
    """
    Drop all frames after the frame of death.

    Args:
        df (pd.DataFrame)
        frame_of_death (int): Frame number indicating death
    """
    return df[df["GlobalFrame"] <= frame_of_death]


def clean_segment_gaps(segment_df, gap_interpolation_limit=10, long_gap_threshold=11):
    """Repair short gaps by interpolation and remove long gaps.

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


def preprocess_file(file, frame_of_death, speed_cap=10, normalize_coords=False):
    """
    Preprocess a single CSV file by applying various cleaning steps.

    Args:
        file (str): Filename to process
        frame_of_death (int): Frame number indicating death
        speed_cap (float): Maximum speed value
        normalize_coords (bool): Whether to normalize coordinates
    """
    df = pd.read_csv(file)
    df = drop_frames_after_death(df, frame_of_death)
    df = cap_speed(df, speed_cap)
    cleaned_segments = []
    for segment_id, segment_df in df.groupby("Segment"):
        segment_df = clean_segment_gaps(segment_df)
        cleaned_segments.append(segment_df)

    cleaned_df = pd.concat(cleaned_segments).reset_index(drop=True)

    if normalize_coords:
        cleaned_df = normalize_coordinates(cleaned_df)

    cleaned_df.to_csv(file, index=False)


def process_all_files(
    treatment, lifespan_summary, speed_cap=10, normalize_coords=False
):
    """
    Preprocess all CSV files in the specified treatment group.

    Args:
        treatment (str): Treatment group
        lifespan_summary (pd.DataFrame): DataFrame containing lifespan summary
        speed_cap (float): Maximum speed value
        normalize_coords (bool): Whether to normalize coordinates
    """
    treatment_dir = os.path.join(PREPROCESSED_DIR, treatment)
    os.makedirs(treatment_dir, exist_ok=True)

    files = glob(os.path.join("data", treatment, "*.csv"))

    for file in files:
        rename_file(os.path.basename(file), treatment, treatment_dir)

    preprocessed_files = glob(os.path.join(treatment_dir, "*.csv"))

    for file in preprocessed_files:
        drop_first_row(os.path.basename(file), treatment_dir)
        add_segment_column(file)
        add_label_column(file, label=treatment == TREATED)
        worm_id = os.path.splitext(os.path.basename(file))[0]
        add_worm_id_column(file, worm_id)

    for file in preprocessed_files:
        worm_id = os.path.splitext(os.path.basename(file))[0]
        frame_of_death = lifespan_summary.loc[
            lifespan_summary["Filename"] == "/" + worm_id, "LifespanInFrames"
        ].values[0]
        preprocess_file(
            file,
            frame_of_death,
            speed_cap=speed_cap,
            normalize_coords=normalize_coords,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Process files with options.")
    parser.add_argument(
        "--speed-cap",
        type=float,
        default=10,
        help="Maximum speed value to cap at (default: 10).",
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
    parser.set_defaults(normalize=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    lifespan_summary = pd.read_csv("data/lifespan_summary.csv")

    process_all_files(
        CONTROL,
        lifespan_summary,
        speed_cap=args.speed_cap,
        normalize_coords=args.normalize,
    )

    process_all_files(
        TREATED,
        lifespan_summary,
        speed_cap=args.speed_cap,
        normalize_coords=args.normalize,
    )
    print("Preprocessing completed.")
