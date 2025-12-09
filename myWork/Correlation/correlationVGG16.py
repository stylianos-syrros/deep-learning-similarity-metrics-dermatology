import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    '(pretrained_avg)block5_conv3', '(pretrained_avg)block5_relu3', '(pretrained_avg)avgpool', 
    '(pretrained_flat)block5_conv3', '(pretrained_flat)block5_relu3', '(pretrained_flat)avgpool', 
    '(pretrained_flat)ReLU', '(pretrained_flat)fc3', '(trained_avg)block5_conv3', 
    '(trained_avg)block5_relu3', '(trained_avg)avgpool', '(trained_flat)block5_conv3', 
    '(trained_flat)block5_relu3',  '(trained_flat)avgpool', '(trained_flat)ReLU', '(trained_flat)fc3'
]
category_columns = ['category', 'color', 'texture']

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Print the results in tabular form
print(f"{'Category':<20} {'Layer':<50} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<50} {result[2]:<20} {result[3]:<20}")

# Find the highest correlation for each category
max_correlations = {}
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col),
        key=lambda x: x[2]
    )
    max_correlations[cat_col] = max_corr

print("\nHighest correlations:")
print(f"{'Category':<20} {'Layer':<50} {'Correlation':<20} {'P-Value':<20}")
for cat_col, result in max_correlations.items():
    print(f"{result[0]:<20} {result[1]:<50} {result[2]:<20} {result[3]:<20}")
