import pandas as pd

# Load the Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances.xlsx"
df = pd.read_excel(file_path)

# Clean the columns by removing the [[]]
columns_to_clean = ['(pretrained_flat)ReLU', '(pretrained_flat)fc3', '(trained_flat)ReLU', '(trained_flat)fc3']

for col in columns_to_clean:
    df[col] = df[col].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))

# Save the cleaned DataFrame to a new Excel file
cleaned_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances.xlsx"
df.to_excel(cleaned_file_path, index=False)

print("Cleaned file saved successfully!")
