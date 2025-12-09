import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# Define the path to the dataset
dataset_path = r"C:\Users\steli\DIPLOMA\bcc\CASE*"

# Get all case folders
case_folders = sorted(glob.glob(dataset_path))

# Split the dataset into 80% train and 20% test with a fixed random seed
train_cases, test_cases = train_test_split(case_folders, test_size=0.3, random_state=42)

# Define the paths for train and test folders
base_path = r"C:\Users\steli\DIPLOMA\bcc"
train_folder_path = os.path.join(base_path, 'final_train_diploma')
test_folder_path = os.path.join(base_path, 'final_test_diploma')

# Create the train and test folders if they do not exist
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(test_folder_path, exist_ok=True)

# Move the cases into their respective folders
for case in train_cases:
    shutil.copytree(case, os.path.join(train_folder_path, os.path.basename(case)))

for case in test_cases:
    shutil.copytree(case, os.path.join(test_folder_path, os.path.basename(case)))

print(f"Folders 'final_train_diploma' and 'final_test_diploma' have been created and populated with cases.")
