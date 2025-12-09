import pandas as pd

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-DISSIMILAR-IMG-NORMALIZED.xlsx"
scores_df = pd.read_excel(file_path)

# Print the column names to identify the issue
print("Column names in the Excel file:", scores_df.columns)

# Attempt to rename the column to a standardized format
if 'CASES' in scores_df.columns:
    scores_df['CASES'] = scores_df['CASES'].str.replace('CASE', '').astype(int)
elif 'cases' in scores_df.columns:
    scores_df['CASES'] = scores_df['cases'].str.replace('CASE', '').astype(int)
else:
    print("The column 'CASES' or 'cases' was not found. Please check the Excel file.")
