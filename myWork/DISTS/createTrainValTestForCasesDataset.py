import os
import shutil
import random

# Ορισμός του seed για σταθερό split
random.seed(42)

# Ορισμός των φακέλων και των διαδρομών
source_dir = "C:/Users/steli/DIPLOMA/bcc"
train_dir = "C:/Users/steli/DIPLOMA/bcc/train"
val_dir = "C:/Users/steli/DIPLOMA/bcc/val"
test_dir = "C:/Users/steli/DIPLOMA/bcc/test"

# Λίστα με όλα τα cases
cases = [f"CASE{str(i).zfill(3)}" for i in range(1, 177)]

# Ανακατεύουμε τα cases για τυχαίο split
random.shuffle(cases)

# Διαχωρισμός σε train, validation και test
train_cases = cases[:123]  # 70% train
val_cases = cases[123:158]  # 20% validation
test_cases = cases[158:]  # 10% test

# Συνάρτηση για να αντιγράφουμε τα cases στους νέους φακέλους
def copy_cases(cases, destination_dir):
    for case in cases:
        source_case_dir = os.path.join(source_dir, case)
        dest_case_dir = os.path.join(destination_dir, case)
        shutil.copytree(source_case_dir, dest_case_dir)

# Δημιουργία φακέλων αν δεν υπάρχουν
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Αντιγραφή των δεδομένων στους νέους φακέλους
copy_cases(train_cases, train_dir)
copy_cases(val_cases, val_dir)
copy_cases(test_cases, test_dir)

print(f"Train: {len(train_cases)} cases, Validation: {len(val_cases)} cases, Test: {len(test_cases)} cases")
