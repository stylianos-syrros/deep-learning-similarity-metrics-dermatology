import pandas as pd

# Load the provided Excel files
categories_df = pd.read_excel('C:\\Users\\steli\\DIPLOMA\\myProgramms\\XLSX\\DISTS_LPIPS_ViT_Categories.xlsx')
scores_df = pd.read_excel('C:\\Users\\steli\\DIPLOMA\\myProgramms\\XLSX\\SCARS-SCORES-RESULT.xlsx')

# Check the columns to identify the correct columns for merging
print("Categories DataFrame Columns:", categories_df.columns)
print("Scores DataFrame Columns:", scores_df.columns)

# Renaming columns if necessary to match the 'case' column for consistency
categories_df.rename(columns={'Folder': 'case'}, inplace=True)

# Merge the dataframes on the 'case' column
merged_df = pd.merge(scores_df, categories_df, on='case')

# Select and rename columns to match the target structure
final_df = merged_df[['case', 'mean_color', 'mean_texture', 'category', 'Category_DISTS', 'Category_LPIPS', 'Category_ViT']]
final_df.columns = ['case', 'color', 'texture', 'category', 'dists', 'lpips', 'vit']

# Save the merged dataframe to a new Excel file
output_path = 'C:\\Users\\steli\\DIPLOMA\\myProgramms\\XLSX\\Merged_Scores_Categories.xlsx'
final_df.to_excel(output_path, index=False)

print(f"Output saved to {output_path}")
