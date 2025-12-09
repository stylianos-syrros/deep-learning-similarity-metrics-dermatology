import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    '(pretrained_avg)block5_conv3', '(pretrained_avg)block5_relu3', '(pretrained_avg)avgpool', 
    '(pretrained_flat)block5_conv3', '(pretrained_flat)block5_relu3', '(pretrained_flat)avgpool', 
    '(pretrained_flat)ReLU', '(pretrained_flat)fc3'#, '(trained_avg)block5_conv3', 
    #'(trained_avg)block5_relu3', '(trained_avg)avgpool', '(trained_flat)block5_conv3', 
    #'(trained_flat)block5_relu3', '(trained_flat)avgpool', '(trained_flat)ReLU', '(trained_flat)fc3'
]
category_columns = ['category']  # Only keeping 'category' column for this analysis

# Calculate the point biserial correlation for each distance column with the 'category' column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])

# Create a DataFrame to store the results
correlation_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])

# Save the correlation results to a new Excel file
output_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\VGG16_Correlation_Category_Results.xlsx"
correlation_df.to_excel(output_file_path, index=False)

print("Correlation results saved to Excel file successfully.")

# Find the highest correlation for the 'category' column
max_correlation = correlation_df.loc[correlation_df['Correlation'].idxmax()]

print("\nHighest correlation:")
print(f"Category: {max_correlation['Category']}, Layer: {max_correlation['Layer']}, "
      f"Correlation: {max_correlation['Correlation']}, P-Value: {max_correlation['P-Value']}")

# Optionally, save the highest correlation details to another Excel sheet
highest_corr_df = pd.DataFrame([max_correlation])
highest_corr_output_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\VGG16_Highest_Correlation_Category.xlsx"
highest_corr_df.to_excel(highest_corr_output_path, index=False)

print("Highest correlation saved to a separate Excel file successfully.")