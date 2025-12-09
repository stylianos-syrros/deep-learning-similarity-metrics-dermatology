import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
average_distance_column = 'Average Distance'
category_columns = ['category']

# Calculate the point biserial correlation for each category column
correlation_results = []
for cat_col in category_columns:
    correlation, p_value = pointbiserialr(df[cat_col], df[average_distance_column])
    correlation_results.append([cat_col, correlation, p_value])

# Create a DataFrame for the results
results_df = pd.DataFrame(correlation_results, columns=['Category', 'Correlation', 'P-Value'])

# Save the results to a new Excel file
output_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\DISTS_Correlation_Results.xlsx"
results_df.to_excel(output_path, index=False)

print("The results were successfully saved to:", output_path)
