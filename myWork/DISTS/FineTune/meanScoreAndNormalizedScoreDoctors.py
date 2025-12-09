import pandas as pd

# Load the Excel file
#file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-IN-DETAIL.xlsx"
#df = pd.read_excel(file_path)

# Calculate the mean score of the doctors for each case
#df['meanScore'] = df[['IB_overall', 'GG_overall', 'MOS_overall', 'SER_overall', 'F_overall', 'H_overall']].mean(axis=1)

# Save the cases and their mean scores into a new Excel file
#output_file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-MEAN.xlsx"
#df[['CASES', 'meanScore']].to_excel(output_file_path, index=False)

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-MEAN.xlsx"
df = pd.read_excel(file_path)

# Normalize the mean scores to the range [0, 0.7]
min_score = df['meanScore'].min()
max_score = df['meanScore'].max()
df['normalizedScore'] = 0.7 * (df['meanScore'] - min_score) / (max_score - min_score)

# Save the updated DataFrame to the same Excel file
df.to_excel(file_path, index=False)