import pandas as pd

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-IN-DETAIL.xlsx"
df = pd.read_excel(file_path)
#print(df.columns.tolist())

# Define the columns for color and texture scores
color_columns = ['IB_color', 'GG_color', 'MOS_color', 'SER_color', 'F_color', 'H_color']
texture_columns = ['ΙΒ_texture', 'GG_texture', 'MOS_texture', 'SER_texture', 'F_texture', 'H_texture']
category_columns = ['IB_overall', 'GG_overall', 'MOS_overall', 'SER_overall', 'F_overall', 'H_overall']


# Calculate the mean scores for each case
df['mean_color_score'] = df[color_columns].mean(axis=1)
df['mean_texture_score'] = df[texture_columns].mean(axis=1)
df['mean_category_score'] = df[category_columns].mean(axis=1)


# Select the relevant columns to save to a new Excel file
output_df = df[['CASES', 'mean_color_score', 'mean_texture_score', 'mean_category_score']]

# Save the output to a new Excel file
output_file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-MEAN-COLOR-TEXTURE.xlsx"
output_df.to_excel(output_file_path, index=False)
output_df.head() 

#import pandas as pd

# Φόρτωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-MEAN-COLOR-TEXTURE.xlsx"
df = pd.read_excel(file_path)

# Λειτουργίες κανονικοποίησης
def normalize_to_range_1_7(series):
    return 1 + 6 * (series - series.min()) / (series.max() - series.min())

# Κανονικοποίηση των στηλών mean_color_score και mean_texture_score στο εύρος 1-7
df['normalized_color_score'] = normalize_to_range_1_7(df['mean_color_score'])
df['normalized_texture_score'] = normalize_to_range_1_7(df['mean_texture_score'])

# Αποθήκευση στο ίδιο αρχείο Excel (προσέχουμε να μην αντικαταστήσουμε άλλα δεδομένα)
df.to_excel(file_path, index=False)
