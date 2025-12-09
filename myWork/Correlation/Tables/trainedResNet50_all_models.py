import pandas as pd
from scipy.stats import pointbiserialr

# Load the dataset into a DataFrame
df = pd.read_excel(r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs(TensorFlow).xlsx")

# Provided column names for distances, ordered with conv5_block3_3_conv first followed by conv5_block3_3_bn
distance_columns = [
    'epoch_01_conv5_block3_3_conv', 'epoch_02_conv5_block3_3_conv', 'epoch_03_conv5_block3_3_conv',
    'epoch_04_conv5_block3_3_conv', 'epoch_05_conv5_block3_3_conv', 'epoch_06_conv5_block3_3_conv',
    'epoch_07_conv5_block3_3_conv', 'epoch_08_conv5_block3_3_conv', 'epoch_09_conv5_block3_3_conv',
    'epoch_10_conv5_block3_3_conv', 'epoch_11_conv5_block3_3_conv', 'epoch_12_conv5_block3_3_conv',
    'epoch_13_conv5_block3_3_conv', 'epoch_14_conv5_block3_3_conv', 'epoch_15_conv5_block3_3_conv',
    'epoch_16_conv5_block3_3_conv', 'epoch_17_conv5_block3_3_conv', 'epoch_18_conv5_block3_3_conv',
    'epoch_19_conv5_block3_3_conv', 'epoch_20_conv5_block3_3_conv', 'epoch_21_conv5_block3_3_conv',
    'epoch_22_conv5_block3_3_conv', 'epoch_23_conv5_block3_3_conv', 'epoch_24_conv5_block3_3_conv',
    'epoch_25_conv5_block3_3_conv',
    'epoch_01_conv5_block3_3_bn', 'epoch_02_conv5_block3_3_bn', 'epoch_03_conv5_block3_3_bn',
    'epoch_04_conv5_block3_3_bn', 'epoch_05_conv5_block3_3_bn', 'epoch_06_conv5_block3_3_bn',
    'epoch_07_conv5_block3_3_bn', 'epoch_08_conv5_block3_3_bn', 'epoch_09_conv5_block3_3_bn',
    'epoch_10_conv5_block3_3_bn', 'epoch_11_conv5_block3_3_bn', 'epoch_12_conv5_block3_3_bn',
    'epoch_13_conv5_block3_3_bn', 'epoch_14_conv5_block3_3_bn', 'epoch_15_conv5_block3_3_bn',
    'epoch_16_conv5_block3_3_bn', 'epoch_17_conv5_block3_3_bn', 'epoch_18_conv5_block3_3_bn',
    'epoch_19_conv5_block3_3_bn', 'epoch_20_conv5_block3_3_bn', 'epoch_21_conv5_block3_3_bn',
    'epoch_22_conv5_block3_3_bn', 'epoch_23_conv5_block3_3_bn', 'epoch_24_conv5_block3_3_bn',
    'epoch_25_conv5_block3_3_bn'
]

# Categories for correlation
category_columns = ['category']

# Calculate point-biserial correlation for each combination of distance and category columns
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])

# Convert the results to a DataFrame and save to Excel
correlation_df = pd.DataFrame(correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])
correlation_df.to_excel(r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\trained_ResNet50_Correlation_All_Models.xlsx", index=False)

# Find and store the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

# Convert the max correlation results to a DataFrame and save to Excel
max_correlation_df = pd.DataFrame(max_correlation_results, columns=['Category', 'Layer', 'Correlation', 'P-Value'])
max_correlation_df.to_excel(r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\trained_ResNet50_Highest_Correlation.xlsx", index=False)

print("Results saved to trained_ResNet50_Correlation_All_Models.xlsx and trained_ResNet50_Highest_Correlation.xlsx")
