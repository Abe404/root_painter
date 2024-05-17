
import matplotlib.pyplot as plt
import argparse
import os
import glob
from datetime import datetime

def find_matching_files(directory, special_string):
    # Find all unique start identifiers for which both train and val files exist
    train_files = glob.glob(os.path.join(directory, '*_train_*.csv'))
    val_files = glob.glob(os.path.join(directory, '*_val_*.csv'))
    
    standard_train_files = []
    standard_val_files = []
    alternative_train_files = []
    alternative_val_files = []
    
    for train_file in train_files:
        start_id = os.path.basename(train_file).split('_')[0]
        val_file = glob.glob(os.path.join(directory, f'{start_id}_val_*.csv'))
        
        if val_file:
            val_file = val_file[0]  # There should be only one match
            if special_string in train_file:
                alternative_train_files.append(train_file)
                alternative_val_files.append(val_file)
            else:
                standard_train_files.append(train_file)
                standard_val_files.append(val_file)
    
    return standard_train_files, standard_val_files, alternative_train_files, alternative_val_files

def calculate_duration(start_time_str, date_time_str):
    start_time = datetime.fromtimestamp(float(start_time_str))
    date_time = datetime.strptime(date_time_str, "%Y-%m-%d-%H:%M:%S")
    duration = (date_time - start_time).total_seconds() / 60.0  # Convert to minutes
    return duration

def read_csv(file_path, x_axis, start_time_str, metric):
    x_values = []
    y_values = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            values = line.strip().split(',')
            if x_axis == 'duration':
                x_value = calculate_duration(start_time_str, values[idx['date_time']])
            else:
                x_value = int(values[idx[x_axis]])
            y_value = float(values[idx[metric]])
            x_values.append(x_value)
            y_values.append(y_value)
    return x_values, y_values

def calculate_mean(x_values, y_values_list):
    mean_y_values = []
    for i in range(len(x_values[0])):
        y_values_at_i = [y_values[i] for y_values in y_values_list if i < len(y_values)]
        mean_y_values.append(sum(y_values_at_i) / len(y_values_at_i))
    return mean_y_values

def plot_metrics(standard_train_files, standard_val_files, alternative_train_files, alternative_val_files, metric, x_axis, output_file):
    plt.figure(figsize=(10, 6))
    
    def plot_file_set(train_files, val_files, train_color, val_color, linestyle):
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
            plt.plot(train_x_values, train_y_values, marker='o', linestyle=linestyle, color=train_color, alpha=0.6)
            plt.plot(val_x_values, val_y_values, marker='o', linestyle=linestyle, color=val_color, alpha=0.6)

        return all_train_x_values, all_train_y_values, all_val_x_values, all_val_y_values

    # Plot standard approach files
    std_train_x, std_train_y, std_val_x, std_val_y = plot_file_set(standard_train_files, standard_val_files, 'blue', 'red', '-')
    
    # Plot alternative approach files
    alt_train_x, alt_train_y, alt_val_x, alt_val_y = plot_file_set(alternative_train_files, alternative_val_files, 'blue', 'red', '--')

    # Calculate and plot mean values for the validation sets
    def plot_mean_val(x_values_list, y_values_list, color, linestyle, label):
        if not y_values_list:
            return

        # Assuming all x_values are the same for all runs
        x_values = x_values_list[0]
        mean_y_values = calculate_mean(x_values_list, y_values_list)
        plt.plot(x_values, mean_y_values, linestyle=linestyle, color=color, label=label)

    # Plot mean values for standard approach
    plot_mean_val(std_val_x, std_val_y, 'black', '-', 'Standard Val Mean')

    # Plot mean values for alternative approach
    plot_mean_val(alt_val_x, alt_val_y, 'black', '--', 'Alternative Val Mean')

    plt.title(f'{metric} over {x_axis.capitalize()}')
    x_axis_label = f'{x_axis.capitalize()} (minutes)' if x_axis == 'duration' else x_axis.capitalize()
    plt.xlabel(x_axis_label)
    plt.ylabel(metric)
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
    parser.add_argument('special_string', type=str, help='Special string to identify alternative approach (e.g., "AdamW")')
    parser.add_argument('output_file', type=str, help='Output file to save the plot as PNG')

    args = parser.parse_args()

    standard_train_files, standard_val_files, alternative_train_files, alternative_val_files = find_matching_files(args.directory, args.special_string)
    plot_metrics(standard_train_files, standard_val_files, alternative_train_files, alternative_val_files, args.metric, args.x_axis, args.output_file)
