import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\LPIPS_Distances_All_Cases_All_Networks.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
lpips_columns = [
    'lpips_alex', 
    'lpips_vgg', 
    'lpips_squeeze'
]
category_columns = ['category']#, 'color', 'texture']

# Calculate the point biserial correlation for each LPIPS column with each category column
correlation_results = []
for cat_col in category_columns:
    for lpips_col in lpips_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[lpips_col])
        correlation_results.append([cat_col, lpips_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Print the results in tabular form
print(f"{'Category':<20} {'LPIPS Column':<20} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<20} {result[2]:<20} {result[3]:<20}")

# Find the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

print("\nHighest correlation for each category:\n")
print(f"{'Category':<20} {'LPIPS Column':<20} {'Correlation':<20} {'P-Value':<20}")
for result in max_correlation_results:
    print(f"{result[0]:<20} {result[1]:<20} {result[2]:<20} {result[3]:<20}")
