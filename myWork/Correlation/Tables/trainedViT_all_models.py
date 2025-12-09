import pandas as pd
from scipy.stats import pointbiserialr
import openpyxl

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ViT_Distances_All_Epochs.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [f'Epoch_{i}' for i in range(1, 26)]
category_columns = ['category']  # Only using 'category'

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Convert correlation results to DataFrame and save to Excel
correlation_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])
correlation_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\trained_ViT_Correlation_All_Models.xlsx"
correlation_df.to_excel(correlation_file_path, index=False)

# Find the highest correlation for each category
max_correlations = []
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col and res[2] != ""),
        key=lambda x: x[2]
    )
    max_correlations.append(max_corr)

# Convert highest correlations to DataFrame and save to Excel
max_correlation_df = pd.DataFrame(max_correlations, columns=['Category', 'Layer', 'Correlation', 'P-Value'])
max_correlation_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\trained_ViT_Highest_Correlation.xlsx"
max_correlation_df.to_excel(max_correlation_file_path, index=False)

print("Correlation results saved to trained_ViT_Correlation_All_Models.xlsx")
print("Highest correlations saved to trained_ViT_Highest_Correlation.xlsx")
