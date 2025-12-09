import os
import lpips
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import pandas as pd

# Ορισμός του βασικού directory με τα cases
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
num_cases = 176

# Λίστα για αποθήκευση των αποτελεσμάτων
results = []

# Λειτουργία για φόρτωση εικόνας και υπολογισμό απόστασης LPIPS
def calculate_lpips_distance(img_path_0, img_path_1, net='alex', version='0.1', use_gpu=False):
    # Φορτώνουμε τις εικόνες
    img0 = lpips.im2tensor(lpips.load_image(img_path_0))
    img1 = lpips.im2tensor(lpips.load_image(img_path_1))
    
    # Κάνουμε resize στις εικόνες σε 224x224
    img0 = TF.resize(img0, (224, 224))
    img1 = TF.resize(img1, (224, 224))
    
    # Αρχικοποίηση του δικτύου LPIPS
    loss_fn = lpips.LPIPS(net=net, version=version)
    
    if use_gpu:
        loss_fn.cuda()
        img0 = img0.cuda()
        img1 = img1.cuda()
    
    # Υπολογισμός της απόστασης
    dist = loss_fn.forward(img0, img1)
    return dist.item()

# Διατρέχουμε όλα τα cases από το CASE001 έως CASE176
for case_num in range(1, num_cases + 1):
    print("Processing case : {case_num}")
    # Ορισμός του directory για το κάθε case
    case_dir = os.path.join(base_dir, f"CASE{case_num:03d}")
    
    # Ορισμός των paths για τις 3 εικόνες
    img_path_0 = os.path.join(case_dir, "0.jpg")
    img_path_1 = os.path.join(case_dir, "1.jpg")
    img_path_2 = os.path.join(case_dir, "2.jpg")
    
    # Υπολογισμός της απόστασης μεταξύ 0-1 και 0-2 για κάθε δίκτυο
    distances_alex = [
        calculate_lpips_distance(img_path_0, img_path_1, net='alex', version='0.1'),
        calculate_lpips_distance(img_path_0, img_path_2, net='alex', version='0.1')
    ]
    distances_vgg = [
        calculate_lpips_distance(img_path_0, img_path_1, net='vgg', version='0.1'),
        calculate_lpips_distance(img_path_0, img_path_2, net='vgg', version='0.1')
    ]
    distances_squeeze = [
        calculate_lpips_distance(img_path_0, img_path_1, net='squeeze', version='0.1'),
        calculate_lpips_distance(img_path_0, img_path_2, net='squeeze', version='0.1')
    ]
    
    # Υπολογισμός της μέσης απόστασης για το κάθε δίκτυο
    mean_distance_alex = np.mean(distances_alex)
    mean_distance_vgg = np.mean(distances_vgg)
    mean_distance_squeeze = np.mean(distances_squeeze)
    
    # Αποθήκευση των αποτελεσμάτων σε μια λίστα
    results.append({
        'Case': f'CASE{case_num:03d}',
        'lpips_alex': mean_distance_alex,
        'lpips_vgg': mean_distance_vgg,
        'lpips_squeeze': mean_distance_squeeze
    })

# Δημιουργία DataFrame και αποθήκευση σε αρχείο Excel
df = pd.DataFrame(results)
output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\LPIPS_Distances_All_Cases_All_Networks.xlsx"
df.to_excel(output_file, index=False)
print(f"\nResults saved to {output_file}")
