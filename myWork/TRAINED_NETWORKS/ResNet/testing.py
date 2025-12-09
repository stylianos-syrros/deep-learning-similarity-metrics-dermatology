import pandas as pd

# Load your Excel file
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs(TensorFlow).xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Remove all columns that contain "Fully_Connected" in their name
df_cleaned = df.loc[:, ~df.columns.str.contains('Fully_Connected')]

# Save the cleaned dataframe back to Excel
df_cleaned.to_excel(r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs(TensorFlow).xlsx", index=False)

print("Columns with 'Fully_Connected' removed successfully.")
