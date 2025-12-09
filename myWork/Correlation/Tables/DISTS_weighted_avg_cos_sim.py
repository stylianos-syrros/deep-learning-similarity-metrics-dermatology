import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf_avg_spatial_weights.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = ['DISTS_tf_avg_spat_weights1', 'DISTS_tf_avg_spat_weights2', 'DISTS_tf_avg_spat_no_img']
category_columns = ['category']  # Uncomment other columns if needed

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])

# Create DataFrame for all correlation results
correlation_results_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Find the highest correlation for each category
max_correlations = []
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col),
        key=lambda x: x[2]
    )
    max_correlations.append(max_corr)

# Create DataFrame for highest correlations
max_correlations_df = pd.DataFrame(max_correlations, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Create an Excel file with two sheets to store the results
output_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\DISTS_tf_weighted_avg_spatial_results.xlsx"
with pd.ExcelWriter(output_path) as writer:
    correlation_results_df.to_excel(writer, sheet_name='All Correlations', index=False)
    max_correlations_df.to_excel(writer, sheet_name='Highest Correlations', index=False)

print(f"Results saved to {output_path}")
