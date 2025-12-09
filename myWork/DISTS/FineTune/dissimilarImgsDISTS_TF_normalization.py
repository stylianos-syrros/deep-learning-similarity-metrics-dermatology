import pandas as pd

# Load the Excel file with DISTS values
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Fine_Tuned_Model_DISTS_tf.xlsx"
#file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf.xlsx"

df = pd.read_excel(file_path)

# Select the column containing DISTS values
dists_column = 'Average Distance'

# Normalize the DISTS values to the range [1, 7]
df['Normalized_DISTS'] = 1 + (df[dists_column] - df[dists_column].min()) * (7 - 1) / (df[dists_column].max() - df[dists_column].min())

# Save the normalized values to a new Excel file
output_file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-DISSIMILAR-IMG-NORMALIZED.xlsx"
df.to_excel(output_file_path, index=False)

print(f'Min value: {df[dists_column].min()}')
print(f'Max value: {df[dists_column].max()}')
print("Saved successfully")