import pandas as pd

# Φόρτωσε το Excel αρχείο
file_path = "C:\\Users\\steli\\DIPLOMA\\myProgramms\\XLSX\\SCARS-SCORES-IN-DETAIL.xlsx"
df = pd.read_excel(file_path)

# Προσθέτουμε τις νέες στήλες για τους γιατρούς
df['IB_overall'] = df['IB_color'] + df['ΙΒ_texture'] - 1
df['GG_overall'] = df['GG_color'] + df['GG_texture'] - 1
df['MOS_overall'] = df['MOS_color'] + df['MOS_texture'] - 1
df['SER_overall'] = df['SER_color'] + df['SER_texture'] - 1
df['F_overall'] = df['F_color'] + df['F_texture'] - 1
df['H_overall'] = df['H_color'] + df['H_texture'] - 1

# Αποθήκευση του αρχείου με τις αλλαγές
output_path = "C:\\Users\\steli\\DIPLOMA\\myProgramms\\XLSX\\SCARS-SCORES-IN-DETAIL.xlsx"
df.to_excel(output_path, index=False)

print(f"Οι αλλαγές αποθηκεύτηκαν στο αρχείο: {output_path}")

