import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\SSIM_Distances.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    'Average SSIM Distance (Resized)'
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
print(f"{'Category':<20} {'Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")
