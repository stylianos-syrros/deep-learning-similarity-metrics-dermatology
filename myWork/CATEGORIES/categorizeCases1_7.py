import pandas as pd

# Διαβάζουμε τα δεδομένα από το Excel αρχείο
file_path = r'C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES.xlsx'
color_df = pd.read_excel(file_path, sheet_name='color')
texture_df = pd.read_excel(file_path, sheet_name='texture')

# Υποθέτουμε ότι οι μέσες τιμές είναι ήδη υπολογισμένες και βρίσκονται στις στήλες 'meanScore' στα δύο φύλλα
color_df['mean_color'] = color_df['meanScore']
texture_df['mean_texture'] = texture_df['meanScore']

# Συνολικό score: άθροισμα των δύο μέσων τιμών
color_df['total_score'] = color_df['mean_color'] + texture_df['mean_texture']

# Κατηγοριοποίηση των scores σύμφωνα με τις τιμές του αθροίσματος
def categorize_score(score):
    return score - 1

color_df['category'] = color_df['total_score'].apply(categorize_score)

# Δημιουργία νέου DataFrame με τις επιθυμητές στήλες
result_df = pd.DataFrame({
    'case': color_df['CASES'],
    'mean_color': color_df['mean_color'],
    'mean_texture': texture_df['mean_texture'],
    'category': color_df['category']
})

# Αποθήκευση των αποτελεσμάτων σε νέο αρχείο Excel
result_file_path = r'C:\Users\steli\DIPLOMA\myProgramms\XLSX\SCARS-SCORES-RESULT.xlsx'
result_df.to_excel(result_file_path, index=False)

print("Results saved to:", result_file_path)
