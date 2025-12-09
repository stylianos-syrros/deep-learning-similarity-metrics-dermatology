import pandas as pd
from scipy.stats import pointbiserialr

# Φόρτωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf.xlsx"
df = pd.read_excel(file_path)

# Επιλογή των σχετικών στηλών
average_distance_column = 'Average Distance'
category_columns = ['category']#, 'color', 'texture']

# Υπολογισμός του point biserial correlation για κάθε στήλη κατηγορίας
correlation_results = []
for cat_col in category_columns:
    correlation, p_value = pointbiserialr(df[cat_col], df[average_distance_column])
    correlation_results.append([cat_col, correlation, p_value])

# Εκτύπωση των αποτελεσμάτων
print(f"{'Category':<20} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    print(f"{result[0]:<20} {result[1]:<20} {result[2]:<20}")
