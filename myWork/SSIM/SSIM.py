import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Φάκελος με τα cases
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
cases = [case for case in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, case))]

# Λειτουργία για την προεπεξεργασία και φόρτωση των εικόνων με αλλαγή μεγέθους
def load_image_with_resize(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Λειτουργία για τον υπολογισμό του SSIM και της απόστασης
def calculate_ssim_distance(imageA, imageB):
    score, _ = ssim(imageA, imageB, full=True)
    distance = 1 - score
    return distance

# Υπολογισμός της μέσης απόστασης για κάθε case και αποθήκευση των αποτελεσμάτων
results = []
for case in cases:
    case_dir = os.path.join(base_dir, case)
    images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]

    # Φορτώνουμε τις εικόνες με αλλαγή μεγέθους
    img1_resized = load_image_with_resize(images[0])
    img2_resized = load_image_with_resize(images[1])
    img3_resized = load_image_with_resize(images[2])

    dist1_resized = calculate_ssim_distance(img1_resized, img2_resized)
    dist2_resized = calculate_ssim_distance(img1_resized, img3_resized)
    avg_distance_resized = (dist1_resized + dist2_resized) / 2

    results.append({
        'Case': case,
        'Average SSIM Distance (Resized)': avg_distance_resized
    })
    print(f"Case {case}: Avg Distance (Resized) = {avg_distance_resized:.4f}")

# Δημιουργία DataFrame και αποθήκευση σε Excel
df = pd.DataFrame(results)
output_file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\SSIM_Distances.xlsx"
df.to_excel(output_file_path, index=False)
print(f"Results saved to {output_file_path}")
