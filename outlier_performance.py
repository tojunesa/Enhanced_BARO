import os
import pandas as pd
import subprocess

# Define datasets and method
datasets = ["re2-tt", "re2-ob", "re2-ss"]
method = "baro"

# List of outlier detection methods to evaluate
outlier_methods = [
    "adaptive_z", "lof", "iqr", "tukey",
    "dbscan", "bayesian", "clipping",
    "winsorization"
]

# Function to run a command and extract accuracy and Avg@5 metrics
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output_lines = result.stdout.split("\n")

    metrics = {
        "CPU": {"Acc@1": 0.0, "Acc@3": 0.0, "Acc@5": 0.0, "Avg@5": 0.0},
        "MEM": {"Acc@1": 0.0, "Acc@3": 0.0, "Acc@5": 0.0, "Avg@5": 0.0},
        "DISK": {"Acc@1": 0.0, "Acc@3": 0.0, "Acc@5": 0.0, "Avg@5": 0.0},
        "SOCKET": {"Acc@1": 0.0, "Acc@3": 0.0, "Acc@5": 0.0, "Avg@5": 0.0},
        "DELAY": {"Acc@1": 0.0, "Acc@3": 0.0, "Acc@5": 0.0, "Avg@5": 0.0},
        "LOSS": {"Acc@1": 0.0, "Acc@3": 0.0, "Acc@5": 0.0, "Avg@5": 0.0}
    }

    for line in output_lines:
        parts = line.split()  # Split line by whitespace
        if len(parts) == 5:  # Ensure the line contains all values
            metric_name, acc1, acc3, acc5, avg5 = parts  # Unpack values

            # Standardize metric name
            metric_name = metric_name.upper()

            if metric_name in metrics:
                metrics[metric_name]["Acc@1"] = float(acc1)
                metrics[metric_name]["Acc@3"] = float(acc3)
                metrics[metric_name]["Acc@5"] = float(acc5)
                metrics[metric_name]["Avg@5"] = float(avg5)

    return metrics

# Store results for all datasets
all_results = {}

for dataset in datasets:
    print(f"\n=== Running baseline for {dataset} ===")
    baseline_command = f"python3 main.py --method {method} --dataset {dataset}"
    baseline_metrics = run_command(baseline_command)

    # Store baseline data separately for each metric
    baseline_data = {
        "Outlier Method": ["Baseline"]
    }
    metric_tables = {sub_metric: {"Outlier Method": ["Baseline"]} for sub_metric in ["Acc@1", "Acc@3", "Acc@5", "Avg@5"]}

    for metric, values in baseline_metrics.items():
        for sub_metric, value in values.items():
            metric_tables[sub_metric][metric] = [value]

    # Run tests for each outlier method
    outlier_results = {sub_metric: {"Outlier Method": []} for sub_metric in ["Acc@1", "Acc@3", "Acc@5", "Avg@5"]}

    for outlier in outlier_methods:
        print(f"Running {outlier} on {dataset}...")
        command = f"python3 main.py --method {method} --dataset {dataset} --outlier {outlier}"
        metrics = run_command(command)

        for sub_metric in outlier_results:
            outlier_results[sub_metric]["Outlier Method"].append(outlier)  # Store only method name
            for metric, values in metrics.items():
                if metric not in outlier_results[sub_metric]:
                    outlier_results[sub_metric][metric] = []
                outlier_results[sub_metric][metric].append(values[sub_metric])

    # Convert to DataFrame
    df_tables = {sub_metric: pd.DataFrame(metric_tables[sub_metric]).set_index("Outlier Method") for sub_metric in ["Acc@1", "Acc@3", "Acc@5", "Avg@5"]}
    df_outliers = {sub_metric: pd.DataFrame(outlier_results[sub_metric]).set_index("Outlier Method") for sub_metric in ["Acc@1", "Acc@3", "Acc@5", "Avg@5"]}

    # Compute percentage difference and average per table
    diff_tables = {}
    for sub_metric in df_tables:
        df_combined = pd.concat([df_tables[sub_metric], df_outliers[sub_metric]])
        df_diff_percent = ((df_outliers[sub_metric] - df_tables[sub_metric].values) / df_tables[sub_metric].values * 100).round(2)
        df_diff_percent["Avg"] = df_diff_percent.mean(axis=1).round(2)
        df_diff_percent.loc["Total Avg"] = df_diff_percent.mean().round(2)

        # Format as percentage for display
        df_percent_final = df_diff_percent.applymap(lambda x: f"{x:.2f}%")

        # Function to add arrows
        def add_arrow(value):
            num = float(value.replace("%", ""))
            if num > 0:
                return f"{value} ↑"  # Red up arrow for improvements
            elif num < 0:
                return f"{value} ↓"  # Green down arrow for degradation
            else:
                return f"{value}"  # No change

        # Apply arrow function
        df_percent_final_arrows = df_percent_final.applymap(add_arrow)

        # Store results for this dataset
        all_results.setdefault(dataset, {})[sub_metric] = {
            "Original Values": df_combined,
            "Percentage Difference": df_percent_final_arrows
        }

# Print all results separately for Acc@1, Acc@3, Acc@5, and Avg@5
for dataset, metrics in all_results.items():
    for sub_metric, results in metrics.items():
        print(f"\n==== Original {sub_metric} Metrics for {dataset} ====\n")
        print(results["Original Values"].to_string())

        print(f"\n==== Performance Comparison of {sub_metric} Against Baseline ({dataset}) ====\n")
        print(results["Percentage Difference"].to_string())

