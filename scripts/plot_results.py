import json
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.plot_utils.presents_results import plot_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot model results.")
    parser.add_argument("--results_file", "-r", type=str, default="results.json",
                        help="Path to the JSON file containing model results.")
    args = parser.parse_args()
    # Example usage
    # Load results from a JSON file
    with open(args.results_file, "r") as f:
        avg_results = json.load(f)

    plot_results(avg_results)