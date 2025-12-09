import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ResNet_Distances_All_Ways.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    'Conv5_x Distance',
    'Average(Spatial Positions) Cosine Distance (Conv5_x)',
    'L2 Pooling Distance',
    'L2 Hanning Pooling Distance',
    'Average Pooling Distance',
    'Fully Connected Layer Distance'
]
category_columns = ['category']

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Create a DataFrame for all correlation results
columns = ["Category", "Layer", "Correlation", "P-Value"]
all_correlations_df = pd.DataFrame(correlation_results, columns=columns)

# Save all correlation results to an Excel file
all_correlations_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\ResNet_All_Correlations.xlsx"
all_correlations_df.to_excel(all_correlations_file_path, index=False)
print(f"All correlation results saved to {all_correlations_file_path}")

# Find the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

# Create a DataFrame for the highest correlation results
highest_correlation_df = pd.DataFrame(max_correlation_results, columns=columns)

# Save the highest correlation result to an Excel file
highest_correlation_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\ResNet_Highest_Correlation.xlsx"
highest_correlation_df.to_excel(highest_correlation_file_path, index=False)
print(f"Highest correlation results saved to {highest_correlation_file_path}")