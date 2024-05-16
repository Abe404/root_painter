#!/bin/bash

# Directory containing the metrics files
METRICS_DIR="./metrics"

# Find all unique start identifiers for which both train and val files exist
start_identifiers=$(ls ${METRICS_DIR}/*_train_*.csv | awk -F'[_]' '{print $1}' | sort | uniq)

# Loop through each start identifier and check for matching train and val files
for start_id in $start_identifiers; do
    echo "Processing start identifier: $start_id"
    train_file=$(ls ${start_id}_train_*.csv 2>/dev/null)
    val_file=$(ls ${start_id}_val_*.csv 2>/dev/null)
    echo "Train file: $train_file"
    echo "Val file: $val_file"

    if [[ -n "$train_file" && -n "$val_file" ]]; then
        # Metric to plot (can be passed as an argument or hardcoded)
        metric=${1:-f1}

        # X-axis to plot (can be passed as an argument or hardcoded)
        x_axis=${2:-epoch}

        # Run the plot_metrics.py script with the matching files
        python plot_metrics.py "$train_file" "$val_file" "$metric" "$x_axis"
    fi
done
