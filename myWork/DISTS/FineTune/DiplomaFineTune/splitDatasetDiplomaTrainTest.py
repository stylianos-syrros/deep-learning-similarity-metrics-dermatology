import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Excel file
excel_file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-IN-DETAIL.xlsx"
df = pd.read_excel(excel_file_path)

# Extract the 'CASES' and 'meanScore_overall' columns
case_names = df['CASES']
categories = df['meanScore_overall']

# Define the path to the dataset
base_path = r"C:\Users\steli\DIPLOMA\bcc"

# Split the dataset into 70% train and 30% test using stratified split based on categories
train_cases, test_cases, _, _ = train_test_split(
    case_names, categories, test_size=0.3, random_state=42, stratify=categories
)

# Define paths for train and test folders
train_folder_path = os.path.join(base_path, 'final_train_diploma')
test_folder_path = os.path.join(base_path, 'final_test_diploma')

# Create the train and test folders if they do not exist
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(test_folder_path, exist_ok=True)

# Function to copy cases to the appropriate folder
def move_cases(case_list, destination_folder):
    for case in case_list:
        case_path = os.path.join(base_path, case)  # Construct the full path to the case folder
        if os.path.exists(case_path):
            shutil.copytree(case_path, os.path.join(destination_folder, case))
        else:
            print(f"Case folder {case} does not exist!")

# Move train cases to the train folder
move_cases(train_cases, train_folder_path)

# Move test cases to the test folder
move_cases(test_cases, test_folder_path)

print(f"Folders 'final_train_diploma' and 'final_test_diploma' have been created and populated with cases from all categories.")
