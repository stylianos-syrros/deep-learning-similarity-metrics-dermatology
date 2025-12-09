import pandas as pd
from scipy.stats import pointbiserialr

# Φορτώστε το αρχείο Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ResNetL2poolingDistancesAllCases.xlsx"
df = pd.read_excel(file_path)

# Επιλέξτε τις σχετικές στήλες
distance_columns = [
    'Average Pooling Distance',
    'L2 Pooling Distance',
    'Average(Spatial Positions) Cosine Distance (Conv5_x)',
    'L2 Hanning Pooling Distance'
]
category_columns = ['category']#, 'color', 'texture']

# Υπολογισμός της point biserial συσχέτισης για κάθε στήλη απόστασης με κάθε κατηγορία
correlation_results = []
for cat_col in category_columns:
    for dist_col in distance_columns:
        correlation, p_value = pointbiserialr(df[cat_col], df[dist_col])
        correlation_results.append([cat_col, dist_col, correlation, p_value])
    # Προσθήκη κενής γραμμής για διαχωρισμό μεταξύ κατηγοριών
    correlation_results.append(["", "", "", ""])

# Εκτύπωση των αποτελεσμάτων σε μορφή πίνακα
print(f"{'Category':<20} {'Distance Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    if result[0] == "":
        print()  # Εκτύπωση κενής γραμμής
    else:
        print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")

# Βρείτε και εκτυπώστε τη γραμμή με τη μεγαλύτερη συσχέτιση για κάθε κατηγορία
max_correlation_results = []
for cat_col in category_columns:
    max_correlation = max((res for res in correlation_results if res[0] == cat_col), key=lambda x: x[2])
    max_correlation_results.append(max_correlation)

print("\nHighest correlation for each category:\n")
print(f"{'Category':<20} {'Distance Layer':<60} {'Correlation':<20} {'P-Value':<20}")
for result in max_correlation_results:
    print(f"{result[0]:<20} {result[1]:<60} {result[2]:<20} {result[3]:<20}")
