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

# Convert the correlation results to a DataFrame
correlation_df = pd.DataFrame(correlation_results, columns=['Category', 'LPIPS Column', 'Correlation', 'P-Value'])

# Save all correlation results to an Excel file
correlation_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\LPIPS_Correlation_All_Cases.xlsx"
correlation_df.to_excel(correlation_file_path, index=False)

# Find the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

# Convert the max correlation results to a DataFrame
max_correlation_df = pd.DataFrame(max_correlation_results, columns=['Category', 'LPIPS Column', 'Correlation', 'P-Value'])

# Save the highest correlation results to an Excel file
max_correlation_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\LPIPS_Highest_Correlation.xlsx"
max_correlation_df.to_excel(max_correlation_file_path, index=False)

print("Correlation tables saved successfully.")
