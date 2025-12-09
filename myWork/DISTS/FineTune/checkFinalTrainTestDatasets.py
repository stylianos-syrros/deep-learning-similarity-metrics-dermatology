import os
import pandas as pd
from collections import defaultdict

# Φόρτωση του αρχείου Excel με τα scores
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-IN-DETAIL.xlsx"
scores_df = pd.read_excel(file_path)

# Καθαρισμός της στήλης 'CASES' για πιθανά κενά
scores_df['CASES'] = scores_df['CASES'].astype(str).str.strip()

# Καθορισμός των φακέλων που θα ελέγξουμε
folders = {
    "train": r"C:\Users\steli\DIPLOMA\bcc\final_train_diploma",
    "test": r"C:\Users\steli\DIPLOMA\bcc\final_test_diploma"
}

# Συνάρτηση για έλεγχο των cases σε κάθε φάκελο και μέτρηση των κατηγοριών
def check_case_categories(folder_path, dataset_type):
    print(f"\nChecking categories for {dataset_type} dataset:")

    # Δημιουργούμε έναν μετρητή για τις κατηγορίες (1 έως 7)
    category_count = defaultdict(int)

    for case_folder in os.listdir(folder_path):
        case_name = case_folder.strip()  # Αφαίρεση κενών από το όνομα του φακέλου

        # Βρίσκουμε την κατηγορία του συγκεκριμένου case από το Excel
        if case_name in scores_df['CASES'].values:
            category = scores_df.loc[scores_df['CASES'] == case_name, 'meanScore_overall'].values[0]
            print(f"{case_name}: Category = {category}")
            category_count[int(category)] += 1  # Ενημερώνουμε τον μετρητή για την κατηγορία
        else:
            print(f"{case_name}: Not found in the Excel file")

    # Επιστροφή των μετρητών για κάθε κατηγορία
    return category_count

# Έλεγχος των φακέλων train και test και αποθήκευση των μετρητών για τις κατηγορίες
train_category_count = check_case_categories(folders['train'], 'train')
test_category_count = check_case_categories(folders['test'], 'test')

# Τελικό print των αποτελεσμάτων
print("\nFinal case counts by category:")

# Εμφάνιση του πλήθους των κατηγοριών για το train
print("\nTrain dataset:")
for category in range(1, 8):  # Οι κατηγορίες είναι από 1 έως 7
    print(f"Category {category}: {train_category_count[category]} cases")

# Εμφάνιση του πλήθους των κατηγοριών για το test
print("\nTest dataset:")
for category in range(1, 8):  # Οι κατηγορίες είναι από 1 έως 7
    print(f"Category {category}: {test_category_count[category]} cases")