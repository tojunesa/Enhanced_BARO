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

# Function to run a command and extract Avg@5 metrics
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output_lines = result.stdout.split("\n")
    metrics = { "CPU": 0.0, "MEM": 0.0, "DISK": 0.0, "SOCKET": 0.0, "DELAY": 0.0, "LOSS": 0.0 }


    for line in output_lines:
        if "Avg@5-CPU:" in line:
            metrics["CPU"] = float(line.split(":")[1].strip())
        elif "Avg@5-MEM:" in line:
            metrics["MEM"] = float(line.split(":")[1].strip())
        elif "Avg@5-DISK:" in line:
            metrics["DISK"] = float(line.split(":")[1].strip())
        elif "Avg@5-SOCKET:" in line:
            metrics["SOCKET"] = float(line.split(":")[1].strip())
        elif "Avg@5-DELAY:" in line:
            metrics["DELAY"] = float(line.split(":")[1].strip())
        elif "Avg@5-LOSS:" in line:
            metrics["LOSS"] = float(line.split(":")[1].strip())

    return metrics

# Store results for all datasets
all_results = {}

for dataset in datasets:
    print(f"\n=== Running baseline for {dataset} ===")
    baseline_command = f"python3 main.py --method {method} --dataset {dataset}"
    baseline_metrics = run_command(baseline_command)

    # Store baseline data
    baseline_data = {
        "Outlier Method": ["Baseline"],
        **{k: [v] for k, v in baseline_metrics.items()}
    }

    # Run tests for each outlier method
    outlier_results = {"Outlier Method": []}
    for outlier in outlier_methods:
        print(f"Running {outlier} on {dataset}...")
        command = f"python3 main.py --method {method} --dataset {dataset} --outlier {outlier}"
        metrics = run_command(command)

        outlier_results["Outlier Method"].append(outlier)  # Store only method name
        for key in metrics:
            if key not in outlier_results:
                outlier_results[key] = []
            outlier_results[key].append(metrics[key])

    # Convert to DataFrame
    df_baseline = pd.DataFrame(baseline_data).set_index("Outlier Method")
    df_outliers = pd.DataFrame(outlier_results).set_index("Outlier Method")

    # Combine baseline and outlier method results into one table
    df_combined = pd.concat([df_baseline, df_outliers])

    # Compute percentage difference
    df_diff_percent = ((df_outliers - df_baseline.values) / df_baseline.values * 100).round(2)

    # Compute average difference per method
    df_diff_percent["Avg"] = df_diff_percent.mean(axis=1).round(2)

    # Compute overall average difference across methods
    total_avg = df_diff_percent.mean().round(2)

    # Append total average as a row
    df_diff_percent.loc["Total Avg"] = total_avg

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
    all_results[dataset] = {"Original Values": df_combined, "Percentage Difference": df_percent_final_arrows}

# Print all results
for dataset, results in all_results.items():
    print(f"\n---- Original Avg@5 Metrics for {dataset} ----\n")
    print(results["Original Values"])
    
    print(f"\n---- Performance Comparison of Outlier Methods Against Baseline ({dataset}) ----\n")
    print(results["Percentage Difference"])

