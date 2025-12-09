import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Φόρτωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-IN-DETAIL.xlsx"
df = pd.read_excel(file_path)

# Ορισμός του seed για σταθερό split
seed = 42

# Δημιουργία του StratifiedShuffleSplit με split 80-20
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

# Διαδρομή του φακέλου με τα αρχικά cases
source_dir = r"C:\Users\steli\DIPLOMA\bcc"
# Διαδρομές για τους φακέλους train_final και test_final
train_dir = r"C:\Users\steli\DIPLOMA\bcc\train_final"
test_dir = r"C:\Users\steli\DIPLOMA\bcc\test_final"

# Δημιουργία των φακέλων train_final και test_final αν δεν υπάρχουν
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Εφαρμογή του split με βάση τη στήλη meanScore_overall
for train_index, test_index in split.split(df, df['meanScore_overall']):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]

# Αντιγραφή των φακέλων των cases στο train_final
for case in strat_train_set['CASES']:
    source_case_folder = os.path.join(source_dir, case)
    destination_case_folder = os.path.join(train_dir, case)
    shutil.copytree(source_case_folder, destination_case_folder)

# Αντιγραφή των φακέλων των cases στο test_final
for case in strat_test_set['CASES']:
    source_case_folder = os.path.join(source_dir, case)
    destination_case_folder = os.path.join(test_dir, case)
    shutil.copytree(source_case_folder, destination_case_folder)

print(f"Τα cases αντιγράφηκαν στους φακέλους:\nTrain: {train_dir}\nTest: {test_dir}")
