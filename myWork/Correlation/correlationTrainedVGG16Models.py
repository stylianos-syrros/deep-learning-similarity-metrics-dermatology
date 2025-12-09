import pandas as pd
from scipy.stats import pointbiserialr

# Load the new Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_VGG16_Distances_All_Epochs.xlsx"
df = pd.read_excel(file_path)

# Select the relevant columns
distance_columns = [
    'Epoch_1_avgpool', 'Epoch_2_avgpool', 'Epoch_3_avgpool', 'Epoch_4_avgpool', 
    'Epoch_5_avgpool', 'Epoch_6_avgpool', 'Epoch_7_avgpool', 'Epoch_8_avgpool', 
    'Epoch_9_avgpool', 'Epoch_10_avgpool', 'Epoch_11_avgpool', 'Epoch_12_avgpool', 
    'Epoch_13_avgpool', 'Epoch_14_avgpool', 'Epoch_15_avgpool', 'Epoch_16_avgpool', 
    'Epoch_17_avgpool', 'Epoch_18_avgpool', 'Epoch_19_avgpool', 'Epoch_20_avgpool', 
    'Epoch_21_avgpool', 'Epoch_22_avgpool', 'Epoch_23_avgpool', 'Epoch_24_avgpool', 
    'Epoch_25_avgpool', 'Epoch_1_classifier.4', 'Epoch_2_classifier.4', 'Epoch_3_classifier.4', 
    'Epoch_4_classifier.4', 'Epoch_5_classifier.4', 'Epoch_6_classifier.4', 'Epoch_7_classifier.4', 
    'Epoch_8_classifier.4', 'Epoch_9_classifier.4', 'Epoch_10_classifier.4', 'Epoch_11_classifier.4', 
    'Epoch_12_classifier.4', 'Epoch_13_classifier.4', 'Epoch_14_classifier.4', 'Epoch_15_classifier.4', 
    'Epoch_16_classifier.4', 'Epoch_17_classifier.4', 'Epoch_18_classifier.4', 'Epoch_19_classifier.4', 
    'Epoch_20_classifier.4', 'Epoch_21_classifier.4', 'Epoch_22_classifier.4', 'Epoch_23_classifier.4', 
    'Epoch_24_classifier.4', 'Epoch_25_classifier.4'
]

category_columns = ['category', 'color', 'texture']

# Calculate the point biserial correlation for each distance column with each category column
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Add an empty row for separation between categories
    correlation_results.append(["", "", "", ""])

# Print the results in tabular form
print(f"{'Category':<20} {'Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Print an empty line
    else:
        print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")

# Find and print the row with the highest correlation for each category
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

print("\nHighest correlation for each category:\n")
print(f"{'Category':<20} {'Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in max_correlation_results:
    print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")
