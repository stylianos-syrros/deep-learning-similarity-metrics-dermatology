import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# Φόρτωση των δεδομένων από το αρχείο Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_LPIPS_VIT_SCORES.xlsx"
df = pd.read_excel(file_path)

# Συνάρτηση για την εφαρμογή του K-means και την απόδοση των κατηγοριών
def apply_kmeans_and_sort(metric_values, n_clusters=7):
    metric_values = metric_values.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(metric_values)
    categories = kmeans.labels_
    
    # Υπολογισμός της μέσης τιμής κάθε cluster
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    
    # Αναδιάταξη των κατηγοριών βάσει των μέσων τιμών
    sorted_categories = np.zeros_like(categories)
    for i, idx in enumerate(sorted_indices):
        sorted_categories[categories == idx] = i + 1
    
    # Υπολογισμός των διαστημάτων κάθε κατηγορίας
    category_ranges = {}
    for i in range(n_clusters):
        indices = sorted_categories == (i + 1)
        min_val = metric_values[indices].min()
        max_val = metric_values[indices].max()
        category_ranges[i + 1] = (min_val, max_val)
    
    return sorted_categories, category_ranges

# Υπολογισμός κατηγοριών και διαστημάτων για κάθε μετρική
categories_dists, ranges_dists = apply_kmeans_and_sort(df['Mean Score DISTS'])
categories_lpips, ranges_lpips = apply_kmeans_and_sort(df['Mean Score LPIPS'])
categories_vit, ranges_vit = apply_kmeans_and_sort(df['trainedViT'])

# Ενημέρωση του DataFrame με τις κατηγορίες
df['Category_DISTS'] = categories_dists
df['Category_LPIPS'] = categories_lpips
df['Category_ViT'] = categories_vit

# Δημιουργία νέου DataFrame με τις επιθυμητές στήλες
result_df = df[['Folder', 'Category_DISTS', 'Category_LPIPS', 'Category_ViT']]

# Αποθήκευση των αποτελεσμάτων σε νέο αρχείο Excel
output_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_LPIPS_ViT_Categories.xlsx"
result_df.to_excel(output_path, index=False)

# Εκτύπωση των διαστημάτων κάθε κατηγορίας
print("DISTS Category Ranges:")
for category, (min_val, max_val) in ranges_dists.items():
    print(f"Category {category}: {min_val:.4f} - {max_val:.4f}")

print("\nLPIPS Category Ranges:")
for category, (min_val, max_val) in ranges_lpips.items():
    print(f"Category {category}: {min_val:.4f} - {max_val:.4f}")

print("\nViT Category Ranges:")
for category, (min_val, max_val) in ranges_vit.items():
    print(f"Category {category}: {min_val:.4f} - {max_val:.4f}")

print("Αποτελέσματα αποθηκεύτηκαν στο αρχείο:", output_path)
