# 
# python plot_metrics.py train_metrics.csv val_metrics.csv f1
#
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(train_file, val_file, metric, x_axis):
    # Read the data from the CSV files
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Add epoch column
    train_df['epoch'] = train_df.index + 1
    val_df['epoch'] = val_df.index + 1

    # Convert duration to minutes if necessary
    if x_axis == 'duration':
        train_df['duration'] = train_df['duration'].astype(float)
        val_df['duration'] = val_df['duration'].astype(float)

    # Extract labels from file names without the .csv extension
    train_label = ' '.join(os.path.basename(train_file).replace('.csv', '').split('_')[1:]) + ' ' + metric
    val_label = ' '.join(os.path.basename(val_file).replace('.csv', '').split('_')[1:]) + ' ' + metric

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_df[x_axis], train_df[metric], marker='o', linestyle='-', label=train_label)
    plt.plot(val_df[x_axis], val_df[metric], marker='o', linestyle='-', label=val_label)
    plt.title(f'{metric} over {x_axis.capitalize()}')
    plt.xlabel(x_axis.capitalize())
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation metrics over epochs or duration.')
    parser.add_argument('train_file', type=str, help='Path to the training metrics file')
    parser.add_argument('val_file', type=str, help='Path to the validation metrics file')
    parser.add_argument('metric', type=str, help='Metric to plot (e.g., f1, precision, recall)')
    parser.add_argument('x_axis', type=str, choices=['epoch', 'duration'], help='X-axis to plot (epoch or duration)')

    args = parser.parse_args()

    plot_metrics(args.train_file, args.val_file, args.metric, args.x_axis)



