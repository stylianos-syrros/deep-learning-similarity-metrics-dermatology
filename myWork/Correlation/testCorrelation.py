import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\best_model_fine_tuned_DISTS_tf.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = ['Average Dist']
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
print(f"{'Category':<20} {'Layer':<35} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<35} {result[2]:<20} {result[3]:<20}")

# Find the highest correlation for each category
max_correlations = {}
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col),
        key=lambda x: x[2]
    )   
    max_correlations[cat_col] = max_corr

print("\nHighest correlations:")
print(f"{'Category':<20} {'Layer':<35} {'Correlation':<20} {'P-Value':<20}")
for cat_col, result in max_correlations.items():
    print(f"{result[0]:<20} {result[1]:<35} {result[2]:<20} {result[3]:<20}")
