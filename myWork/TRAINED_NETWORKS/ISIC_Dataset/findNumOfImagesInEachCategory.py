import pandas as pd

# Φόρτωση του αρχείου CSV
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\TRAINED_NETWORKS\ISIC_Dataset\ISIC_2019_Training_GroundTruth.csv"
df = pd.read_csv(file_path)

# Αφαίρεση της στήλης που περιέχει τα ονόματα των εικόνων
image_column = df.columns[0]
categories = df.columns[1:]

# Υπολογισμός και εκτύπωση του αριθμού των εικόνων σε κάθε κατηγορία
for category in categories:
    count = df[category].sum()
    print(f'Category {category}: {count} images')
