import pandas as pd
from scipy.stats import pointbiserialr

# Load the new Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Fine_Tuned_Model_DISTS_tf_all_epochs.xlsx" 
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [f'epoch_{i}_dists' for i in range(1, 26)]
category_columns = ['category', 'color', 'texture']

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Create DataFrame for all correlation results
correlation_results_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Save all correlation results to an Excel file
output_path_all = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\Fine_Tuned_Model_DISTS_All_Correlation_Results.xlsx"
correlation_results_df.to_excel(output_path_all, index=False)
print(f"All correlation results saved to {output_path_all}")

# Find and store the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

# Create DataFrame for highest correlations
max_correlation_results_df = pd.DataFrame(max_correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Save highest correlations to an Excel file
output_path_max = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\Fine_Tuned_Model_DISTS_Highest_Correlation_Results.xlsx"
max_correlation_results_df.to_excel(output_path_max, index=False)
print(f"Highest correlation results saved to {output_path_max}")
