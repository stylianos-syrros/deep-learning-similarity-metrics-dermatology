# Based on the provided column names, let's set up the correlation calculation.

import pandas as pd
from scipy.stats import pointbiserialr

# Assuming the dataset has been loaded into a DataFrame
# Below is the structure for calculating correlation between the required columns

# Replace with the actual dataframe loading step
df = pd.read_excel(r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs(TensorFlow).xlsx") 

# Provided column names
distance_columns = [
    'epoch_01_conv5_block3_3_conv', 'epoch_01_conv5_block3_3_bn', 'epoch_02_conv5_block3_3_conv',
    'epoch_02_conv5_block3_3_bn', 'epoch_03_conv5_block3_3_conv', 'epoch_03_conv5_block3_3_bn',
    'epoch_04_conv5_block3_3_conv', 'epoch_04_conv5_block3_3_bn', 'epoch_05_conv5_block3_3_conv',
    'epoch_05_conv5_block3_3_bn', 'epoch_06_conv5_block3_3_conv', 'epoch_06_conv5_block3_3_bn',
    'epoch_07_conv5_block3_3_conv', 'epoch_07_conv5_block3_3_bn', 'epoch_08_conv5_block3_3_conv',
    'epoch_08_conv5_block3_3_bn', 'epoch_09_conv5_block3_3_conv', 'epoch_09_conv5_block3_3_bn',
    'epoch_10_conv5_block3_3_conv', 'epoch_10_conv5_block3_3_bn', 'epoch_11_conv5_block3_3_conv',
    'epoch_11_conv5_block3_3_bn', 'epoch_12_conv5_block3_3_conv', 'epoch_12_conv5_block3_3_bn',
    'epoch_13_conv5_block3_3_conv', 'epoch_13_conv5_block3_3_bn', 'epoch_14_conv5_block3_3_conv',
    'epoch_14_conv5_block3_3_bn', 'epoch_15_conv5_block3_3_conv', 'epoch_15_conv5_block3_3_bn',
    'epoch_16_conv5_block3_3_conv', 'epoch_16_conv5_block3_3_bn', 'epoch_17_conv5_block3_3_conv',
    'epoch_17_conv5_block3_3_bn', 'epoch_18_conv5_block3_3_conv', 'epoch_18_conv5_block3_3_bn',
    'epoch_19_conv5_block3_3_conv', 'epoch_19_conv5_block3_3_bn', 'epoch_20_conv5_block3_3_conv',
    'epoch_20_conv5_block3_3_bn', 'epoch_21_conv5_block3_3_conv', 'epoch_21_conv5_block3_3_bn',
    'epoch_22_conv5_block3_3_conv', 'epoch_22_conv5_block3_3_bn', 'epoch_23_conv5_block3_3_conv',
    'epoch_23_conv5_block3_3_bn', 'epoch_24_conv5_block3_3_conv', 'epoch_24_conv5_block3_3_bn',
    'epoch_25_conv5_block3_3_conv', 'epoch_25_conv5_block3_3_bn', 'conv5_block3_out_cosine',
    'conv5_block3_out_avg_cosine', 'global_average_pooling2d'
]

# Categories for correlation
category_columns = ['category']#'color']#, 'texture', 'category']

# Calculate point-biserial correlation for each combination of distance columns and category columns
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Print the results in tabular form
print(f"{'Category':<20} {'Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")

# Find and print the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

print("\nHighest correlation for each category:\n")
print(f"{'Category':<20} {'Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in max_correlation_results:
    print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")

