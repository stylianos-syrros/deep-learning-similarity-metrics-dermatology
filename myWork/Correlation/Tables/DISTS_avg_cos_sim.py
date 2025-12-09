import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf_avg_spatial.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = ['Average Distance']
category_columns = ['category']

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Create a DataFrame for the results
results_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Save the results to a new Excel file
output_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\DISTS_avg_cos_Correlation_Results.xlsx"
results_df.to_excel(output_path, index=False)

print("The correlation results have been successfully saved to:", output_path)
