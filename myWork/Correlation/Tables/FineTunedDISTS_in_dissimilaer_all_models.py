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

# Δημιουργία DataFrame για όλα τα αποτελέσματα συσχέτισης
correlation_results_df = pd.DataFrame(correlation_results, columns=['Category', 'Epoch', 'Correlation', 'P-Value'])

# Εύρεση του epoch με τη μεγαλύτερη συσχέτιση για την κατηγορία
max_correlation_result = max(correlation_results, key=lambda x: x[2])
max_correlation_results_df = pd.DataFrame([max_correlation_result], columns=['Category', 'Epoch', 'Correlation', 'P-Value'])

# Αποθήκευση των αποτελεσμάτων σε δύο αρχεία Excel
output_path_all = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\Diploma_Fine_Tuned_Model_DISTS_All_Correlation_Results.xlsx"
output_path_max = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Tables\Diploma_Fine_Tuned_Model_DISTS_Highest_Correlation_Results.xlsx"

correlation_results_df.to_excel(output_path_all, index=False)
max_correlation_results_df.to_excel(output_path_max, index=False)

print(f"Results saved to {output_path_all} and {output_path_max}")
