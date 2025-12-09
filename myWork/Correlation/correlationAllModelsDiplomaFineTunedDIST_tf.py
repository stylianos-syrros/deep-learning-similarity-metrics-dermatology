import pandas as pd
from scipy.stats import pointbiserialr

# Φόρτωση του νέου αρχείου Excel με τα φιλτραρισμένα δεδομένα
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Diploma_Only_Test_Fine_Tuned_Model_DISTS_tf_all_epochs.xlsx"
df = pd.read_excel(file_path)

# Επιλογή των σχετικών στηλών: οι στήλες των εποχών (epochs) και η στήλη 'category'
distance_columns = [f'epoch_{i}_dists' for i in range(1, 26)]
category_column = 'category'  # Υποθέτουμε ότι υπάρχει στήλη 'category' στο αρχείο

# Υπολογισμός της point-biserial συσχέτισης για κάθε στήλη από τις εποχές με την κατηγορία
correlation_results = []
for dist_col in distance_columns:
    correlation, p_value = pointbiserialr(df[category_column], df[dist_col])
    correlation_results.append([category_column, dist_col, correlation, p_value])

# Εκτύπωση των αποτελεσμάτων με μορφή πίνακα
print(f"{'Category':<20} {'Epoch':<30} {'Correlation':<20} {'P-Value':<20}")
for result in correlation_results:
    print(f"{result[0]:<20} {result[1]:<30} {result[2]:<20} {result[3]:<20}")

# Αν θέλεις να βρεις το epoch με τη μεγαλύτερη συσχέτιση για την κατηγορία:
max_correlation_result = max(correlation_results, key=lambda x: x[2])

print("\nEpoch with the highest correlation for 'category':")
print(f"Epoch: {max_correlation_result[1]}, Correlation: {max_correlation_result[2]}, P-Value: {max_correlation_result[3]}")
