import os
import pandas as pd

# Φόρτωση του αρχείου Excel με τα CASES και τις τιμές από τα epochs
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Diploma_Fine_Tuned_Model_DISTS_tf_all_epochs.xlsx"
df = pd.read_excel(file_path)

# Ανάγνωση των φακέλων από το final_test_diploma
test_folder_path = r"C:\Users\steli\DIPLOMA\bcc\final_test_diploma"
test_cases = [folder_name for folder_name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, folder_name))]

# Καθαρισμός των ονομάτων των φακέλων ώστε να ταιριάζουν με τα δεδομένα στο Excel (π.χ. CASE001)
test_cases_cleaned = [case.strip() for case in test_cases]

# Φιλτράρισμα του DataFrame για να κρατήσει μόνο τα rows που περιέχουν τα cases από το final_test_diploma
df_filtered = df[df['Case'].isin(test_cases_cleaned)]  # Assuming 'CASES' column contains the case names

# Αποθήκευση του φιλτραρισμένου DataFrame σε νέο αρχείο Excel
filtered_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Diploma_Only_Test_Fine_Tuned_Model_DISTS_tf_all_epochs.xlsx"
df_filtered.to_excel(filtered_file_path, index=False)

print(f"Filtered data saved to {filtered_file_path}")
