
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(train_file, val_file, metric):
    # Read the data from the CSV files
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Add epoch column
    train_df['epoch'] = train_df.index + 1
    val_df['epoch'] = val_df.index + 1

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['epoch'], train_df[metric], marker='o', linestyle='-', label='Train ' + metric)
    plt.plot(val_df['epoch'], val_df[metric], marker='o', linestyle='-', label='Validation ' + metric)
    plt.title(f'{metric} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation metrics over epochs.')
    parser.add_argument('train_file', type=str, help='Path to the training metrics file')
    parser.add_argument('val_file', type=str, help='Path to the validation metrics file')
    parser.add_argument('metric', type=str, help='Metric to plot (e.g., f1, precision, recall)')

    args = parser.parse_args()

    plot_metrics(args.train_file, args.val_file, args.metric)


# 
# python plot_metrics.py train_metrics.csv val_metrics.csv f1
#
