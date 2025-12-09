import pandas as pd
from scipy.stats import pearsonr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DifferentVGG16_layers.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    'Distance features.28',
    'Distance features.29',
    'Distance features.30',
    'Distance features.allChannels',
    'Distance features.block5',
    'Distance features.avgpool'
]
category_columns = ['color', 'texture', 'category']

# Calculate the Pearson correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pearsonr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Print the results in tabular form with increased spacing
print(f"{'Category':<20} {'Layer':<30} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<30} {result[2]:<20} {result[3]:<20}")

# Find the highest correlation for each category
max_correlations = {}
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col),
        key=lambda x: x[2]
    )
    max_correlations[cat_col] = max_corr

print("\nHighest correlations:")
print(f"{'Category':<20} {'Layer':<30} {'Correlation':<20} {'P-Value':<20}")
for cat_col, result in max_correlations.items():
    print(f"{result[0]:<20} {result[1]:<30} {result[2]:<20} {result[3]:<20}")
