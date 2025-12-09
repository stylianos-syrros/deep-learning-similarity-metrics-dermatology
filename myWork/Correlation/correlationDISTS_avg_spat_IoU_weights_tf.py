import pandas as pd
from scipy.stats import pointbiserialr

# Φόρτωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf_avg_spatial_IoU_weights.xlsx"
df = pd.read_excel(file_path)

# Επιλογή των σχετικών στηλών
distance_columns = ['DISTS_tf_avg_spat_IoU_weights1', 'DISTS_tf_avg_spat_IoU_weights2']
category_columns = ['color', 'texture', 'category']

# Υπολογισμός του point biserial correlation για κάθε στήλη απόστασης με κάθε στήλη κατηγορίας
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Προσθήκη κενής γραμμής για διαχωρισμό ανάμεσα στις κατηγορίες
    correlation_results.append(["", "", "", ""])

# Εκτύπωση αποτελεσμάτων σε μορφή πίνακα
print(f"{'Category':<20} {'Layer':<35} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Εκτύπωση κενής γραμμής
    else:
        print(f"{result[0]:<20} {result[1]:<35} {result[2]:<20} {result[3]:<20}")

# Εύρεση της υψηλότερης συσχέτισης για κάθε κατηγορία
max_correlations = {}
for cat_col in category_columns:
    max_corr = max(
        (res for res in correlation_results if res[0] == cat_col),
        key=lambda x: x[2]
    )   
    max_correlations[cat_col] = max_corr

print("\nHighest correlations:")
print(f"{'Category':<20} {'Layer':<35} {'Correlation':<20} {'P-Value':<20}")
for cat_col, result in max_correlations.items():
    print(f"{result[0]:<20} {result[1]:<35} {result[2]:<20} {result[3]:<20}")
