import matplotlib.pyplot as plt
import argparse
import os
import glob
from datetime import datetime
import math

def find_matching_files(directory, term1, term2):
    # Find all unique start identifiers for which both train and val files exist
    train_files = glob.glob(os.path.join(directory, '*_train_*.csv'))
    val_files = glob.glob(os.path.join(directory, '*_val_*.csv'))
    
    term1_train_files = []
    term1_val_files = []
    term2_train_files = []
    term2_val_files = []
    
    for train_file in train_files:
        start_id = os.path.basename(train_file).split('_')[0]
        val_file = glob.glob(os.path.join(directory, f'{start_id}_val_*.csv'))
        
        if val_file:
            val_file = val_file[0]  # There should be only one match
            if term2 in train_file:
                term2_train_files.append(train_file)
                term2_val_files.append(val_file)
            elif term1 in train_file:
                term1_train_files.append(train_file)
                term1_val_files.append(val_file)
    
    return term1_train_files, term1_val_files, term2_train_files, term2_val_files

def calculate_duration(start_time_str, date_time_str):
    start_time = datetime.fromtimestamp(float(start_time_str))
    date_time = datetime.strptime(date_time_str, "%Y-%m-%d-%H:%M:%S")
    duration = (date_time - start_time).total_seconds() / 60.0  # Convert to minutes
    return duration

def read_csv(file_path, x_axis, start_time_str, metric):
    print('reading', file_path)
    x_values = []
    y_values = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        idx = {col: i for i, col in enumerate(header)}
        for i, line in enumerate(f):
            values = line.strip().split(',')
            if x_axis == 'duration':
                x_value = calculate_duration(start_time_str, values[idx['date_time']])
            elif x_axis == 'epoch':
                x_value = i + 1
            else:
                x_value = int(values[idx[x_axis]])
            y_value = float(values[idx[metric]])
            x_values.append(x_value)
            y_value = 0 if math.isnan(y_value) else y_value 
            y_values.append(y_value)
    return x_values, y_values

def calculate_mean(x_values, y_values_list):
    mean_y_values = []
    for i in range(len(x_values[0])):
        y_values_at_i = [y_values[i] for y_values in y_values_list if i < len(y_values)]
        mean_y_values.append(sum(y_values_at_i) / len(y_values_at_i))
    return mean_y_values

def plot_metrics(term1_train_files, term1_val_files, term2_train_files, term2_val_files, metric, x_axis, output_file, display_mode, term1_label, term2_label):
    plt.figure(figsize=(12, 8))
    added_labels = {}

    def plot_file_set(train_files, val_files, train_color, val_color, linestyle, label_prefix):
        all_train_x_values, all_train_y_values = [], []
        all_val_x_values, all_val_y_values = [], []

        for train_file, val_file in zip(train_files, val_files):
            start_time_str = os.path.basename(train_file).split('_')[0]

            # Read data
            train_x_values, train_y_values = read_csv(train_file, x_axis, start_time_str, metric)
            val_x_values, val_y_values = read_csv(val_file, x_axis, start_time_str, metric)

            # Store values for mean calculation
            all_train_x_values.append(train_x_values)
            all_train_y_values.append(train_y_values)
            all_val_x_values.append(val_x_values)
            all_val_y_values.append(val_y_values)

            # Plotting individual runs
            train_label = f'{label_prefix} Train'
            val_label = f'{label_prefix} Val'
            plt.plot(train_x_values, train_y_values, marker='o', linestyle=linestyle, color=train_color, alpha=0.6, linewidth=1, label=train_label if train_label not in added_labels else "")
            plt.plot(val_x_values, val_y_values, marker='o', linestyle=linestyle, color=val_color, alpha=0.6, linewidth=1, label=val_label if val_label not in added_labels else "")

            # Mark labels as added
            added_labels[train_label] = True
            added_labels[val_label] = True

        return all_train_x_values, all_train_y_values, all_val_x_values, all_val_y_values

    def plot_mean_val(x_values_list, y_values_list, color, linestyle, label):
        if not y_values_list:
            return
        x_values = x_values_list[0]
        mean_y_values = calculate_mean(x_values_list, y_values_list)
        plt.plot(x_values, mean_y_values, linestyle=linestyle, color=color, label=label, linewidth=2)

    if display_mode in ['both', term1_label]:
        term1_train_x, term1_train_y, term1_val_x, term1_val_y = plot_file_set(term1_train_files, term1_val_files, 'blue', 'red', '-', term1_label)
        plot_mean_val(term1_val_x, term1_val_y, 'black', '-', f'{term1_label} Val Mean')

    if display_mode in ['both', term2_label]:
        term2_train_x, term2_train_y, term2_val_x, term2_val_y = plot_file_set(term2_train_files, term2_val_files, 'purple', 'orange', '--', term2_label)
        plot_mean_val(term2_val_x, term2_val_y, 'black', '--', f'{term2_label} Val Mean')

    plt.title(f'{metric} over {x_axis.capitalize()}')
    x_axis_label = f'{x_axis.capitalize()} (minutes)' if x_axis == 'duration' else x_axis.capitalize()
    plt.xlabel(x_axis_label)
    plt.ylabel(metric)
    plt.ylim([0, 1])  # Set y-axis limit for consistent comparison
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0])
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PNG file
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation metrics over epochs or duration.')
    parser.add_argument('directory', type=str, help='Directory containing the metrics files')
    parser.add_argument('metric', type=str, help='Metric to plot (e.g., f1, precision, recall)')
    parser.add_argument('x_axis', type=str, choices=['epoch', 'duration'], help='X-axis to plot (epoch or duration)')
    parser.add_argument('term1_string', type=str, help='String to identify first term approach (e.g., "baseline")')
    parser.add_argument('term2_string', type=str, help='String to identify second term approach (e.g., "adamw")')
    parser.add_argument('output_file', type=str, help='Output file to save the plot as PNG')
    parser.add_argument('display_mode', type=str)

    args = parser.parse_args()

    term1_train_files, term1_val_files, term2_train_files, term2_val_files = find_matching_files(args.directory, args.term1_string, args.term2_string)
    plot_metrics(term1_train_files, term1_val_files, term2_train_files, term2_val_files, args.metric, args.x_axis, args.output_file, args.display_mode, args.term1_string, args.term2_string)
