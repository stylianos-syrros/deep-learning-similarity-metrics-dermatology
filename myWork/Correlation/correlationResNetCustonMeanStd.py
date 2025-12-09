import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ResNet_Distances_CustomMeanStd.xlsx"
df = pd.read_excel(file_path)

# Print the column names to check if they match
print("Columns in the DataFrame:", df.columns)

# Select the relevant columns based on the actual names in your DataFrame
distance_columns = [
    'Conv5_x Distance',
    'Average Pooling Distance',
    'Fully Connected Layer Distance'
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
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")

# Find and print the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
