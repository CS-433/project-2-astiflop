import numpy as np
import pandas as pd
import os

CONTROL = "TERBINAFINE- (control)"
TREATED = "TERBINAFINE+"
PREPROCESSED_DIR = "preprocessed_data/"
CLASSIFIER_DIR = "preprocessed_data_for_classifier/"


def convert_wormdf_to_segmentsdf(worm_df, label):
    """
    Considering that we already split the data into segments, this function converts a worm-level
    df into a segments-level df by computing
    - Age at the segment : Segment * 6h (in hours)
    - Mean Speed in the segment
    - Median Speed in the segment
    - Net displacement in the segment : distance between first and last point in the segment
    - Tortuosity in the segment : total distance traveled / net displacement
    Args:
        worm_df (pd.DataFrame): DataFrame containing data for a single worm with segments.
        label (str): Label for the worm (e.g., treatment group)
    """
    segments_data = []
    grouped = worm_df.groupby("Segment")
    for segment, segment_df in grouped:
        age_hours = segment * 6
        mean_speed = segment_df["Speed"].mean()
        median_speed = segment_df["Speed"].median()
        if len(segment_df) < 2:
            net_displacement = 0
            tortuosity = np.nan
        else:
            start_pos = segment_df.iloc[0][["X", "Y"]].values
            end_pos = segment_df.iloc[-1][["X", "Y"]].values
            net_displacement = np.linalg.norm(end_pos - start_pos)
            total_distance = (
                segment_df[["X", "Y"]]
                .diff()
                .dropna()
                .apply(lambda row: np.linalg.norm(row), axis=1)
                .sum()
            )
            tortuosity = (
                total_distance / net_displacement if net_displacement != 0 else 0
            )

        segments_data.append(
            {
                "Segment": segment,
                "Age_hours": age_hours,
                "Mean_Speed": mean_speed,
                "Median_Speed": median_speed,
                "Net_Displacement": net_displacement,
                "Tortuosity": tortuosity,
                "Terbinafine": label,
            }
        )
    segments_df = pd.DataFrame(segments_data).fillna(0)
    return segments_df


def process_worms_for_classifier(
    preprocessed_dir, treatment, classifier_dir, lifespan_summary
):
    """
    Process all worm CSV files in the preprocessed directory to create a segments-level DataFrame
    suitable for classifier training. The resulting DataFrame is saved as a CSV file.

    Args:
        preprocessed_dir (str): Directory containing preprocessed worm CSV files.
        treatment (str): Treatment group name.
        classifier_dir (str): Directory to save the classifier-ready CSV file.
        lifespan_summary (pd.DataFrame): DataFrame containing lifespan summary with worm IDs and labels.
    """
    all_segments_df = []
    treatment_dir = os.path.join(preprocessed_dir, treatment)
    for file_name in os.listdir(treatment_dir):
        if file_name.endswith(".csv"):
            worm_id = file_name[:-4]
            worm_df = pd.read_csv(os.path.join(treatment_dir, file_name))
            label_row = lifespan_summary[lifespan_summary["Filename"] == "/" + worm_id]
            if not label_row.empty:
                label = label_row["Terbinafine"].values[0]
                segments_df = convert_wormdf_to_segmentsdf(worm_df, label)
                segments_df["Worm_ID"] = worm_id
                all_segments_df.append(segments_df)

    if all_segments_df:
        final_df = pd.concat(all_segments_df, ignore_index=True)
        os.makedirs(classifier_dir, exist_ok=True)
        final_df.to_csv(
            os.path.join(classifier_dir, f"{treatment}_segments_data.csv"), index=False
        )


if __name__ == "__main__":
    lifespan_summary = pd.read_csv("data/lifespan_summary.csv")
    process_worms_for_classifier(
        PREPROCESSED_DIR, CONTROL, CLASSIFIER_DIR, lifespan_summary
    )
    process_worms_for_classifier(
        PREPROCESSED_DIR, TREATED, CLASSIFIER_DIR, lifespan_summary
    )
    print("Classifier data processing completed.")
