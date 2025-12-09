import re

# Function to read and process the log file
def analyze_log_file(log_file_path):
    # Initialize variables to store the best metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0

    # Initialize variables to track epochs with best metrics
    best_val_loss_epoch = -1
    best_val_acc_epoch = -1

    # Read the log file
    with open(log_file_path, 'r') as file:
        log_data = file.readlines()

    # Process the log data
    for idx, line in enumerate(log_data):
        # Extract validation loss and accuracy from the same line
        val_metrics_match = re.search(r"Val Loss: ([\d.]+) Acc: ([\d.]+)", line)
        if val_metrics_match:
            val_loss = float(val_metrics_match.group(1))
            val_acc = float(val_metrics_match.group(2))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = idx // 4 + 1  # Calculate epoch from index
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_acc_epoch = idx // 4 + 1  # Calculate epoch from index

    return best_val_loss, best_val_loss_epoch, best_val_acc, best_val_acc_epoch

# User input for log file path
log_file_path = input("Enter the path to the log file: ")

# Analyze the log file
best_val_loss, best_val_loss_epoch, best_val_acc, best_val_acc_epoch = analyze_log_file(log_file_path)

# Print the results
print(f"Best Val Loss: {best_val_loss} at epoch {best_val_loss_epoch}")
print(f"Best Val Accuracy: {best_val_acc} at epoch {best_val_acc_epoch}")