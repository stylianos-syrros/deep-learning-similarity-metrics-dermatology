import pandas as pd
from scipy.stats import pointbiserialr

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs(PyTorch).xlsx"
df = pd.read_excel(file_path)

# Convert categorical columns to numeric
for cat_col in ['category', 'color', 'texture']:
    df[cat_col] = pd.factorize(df[cat_col])[0]

# Clean up the distance columns to remove extra brackets and convert to float
for dist_col in df.columns:
    if dist_col.startswith('Epoch_'):
        df[dist_col] = df[dist_col].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

# Select the relevant columns
distance_columns = [
    f'Epoch_{i}_layer3_cosine' for i in range(1, 26)
] + [
    f'Epoch_{i}_avgpool' for i in range(1, 26)
] + [
    f'Epoch_{i}_fc' for i in range(1, 26)
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
min_correlation_results = []
for cat_col in category_columns:
    min_correlation = min((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    min_correlation_results.append(min_correlation)

print("\nHighest correlation for each category:\n")
print(f"{'Category':<20} {'Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in min_correlation_results:
    print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")
