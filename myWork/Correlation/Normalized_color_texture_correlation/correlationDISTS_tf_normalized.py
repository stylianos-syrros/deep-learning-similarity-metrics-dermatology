import pandas as pd
from scipy.stats import pointbiserialr

# Φόρτωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\NORMALIZED_COLOR_TEXTURE\normalized_DISTS_tf.xlsx"
df = pd.read_excel(file_path)

# Επιλογή των σχετικών στηλών
average_distance_column = 'Average Distance'
#category_columns = ['mean_category_score', 'normalized_color_score', 'normalized_texture_score']
category_columns = ['category', 'normalized_color_score', 'normalized_texture_score']


# Υπολογισμός του point biserial correlation για κάθε στήλη κατηγορίας
correlation_results = []
for cat_col in category_columns:
    correlation, p_value = pointbiserialr(df[cat_col], df[average_distance_column])
    correlation_results.append([cat_col, correlation, p_value])

# Εκτύπωση των αποτελεσμάτων
print(f"{'Category':<40} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    print(f"{result[0]:<40} {result[1]:<20} {result[2]:<20}")
