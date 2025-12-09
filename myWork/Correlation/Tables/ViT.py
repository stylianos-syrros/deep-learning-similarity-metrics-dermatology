import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ViT_Distances.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    'pretrainedViTEmb', 'pretrainedViTCLS'#,
    #'trainedViTEmb', 'trainedViTCLS'
]
category_columns = ['category']#, 'color', 'texture']

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Convert the results to a DataFrame
correlation_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Save the correlation results to an Excel file
output_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\ViT_Correlation_Results.xlsx"
correlation_df.to_excel(output_file_path, index=False)

print(f"Correlation results saved to {output_file_path}")

# Find the highest correlation for each category
max_correlations = {}
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col),
        key=lambda x: x[2]
    )
    max_correlations[cat_col] = max_corr

# Convert max correlation results to a DataFrame
max_correlation_results = pd.DataFrame(
    list(max_correlations.values()), 
    columns=['Category', 'Layer', 'Correlation', 'P-Value']
)

# Save the highest correlation results to another Excel file
max_output_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\ViT_Highest_Correlation.xlsx"
max_correlation_results.to_excel(max_output_file_path, index=False)

print(f"Highest correlation results saved to {max_output_file_path}")
