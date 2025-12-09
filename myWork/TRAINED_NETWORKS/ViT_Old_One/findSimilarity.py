import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Παράκαμψη της προειδοποίησης για symlinks
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Καταστολή συγκεκριμένης προειδοποίησης
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

# Ελέγχει αν η GPU είναι διαθέσιμη και θέτει την συσκευή
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT απαιτεί σταθερό μέγεθος εισόδου
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.7438, 0.5865, 0.5869], std=[0.0804, 0.1076, 0.1202])
    transforms.Normalize(mean=[0.65886277, 0.45850426, 0.41027424], std=[0.14796351, 0.14156736, 0.13876683])
])

# Φόρτωση του προεκπαιδευμένου μοντέλου ViT και του best model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # num_classes=0 to get features

# Path to the best model
best_model_path = r"D:\Diploma\models\best_model.pth"
state_dict = torch.load(best_model_path)
# Αφαίρεση των παραμέτρων head από το state_dict
for key in list(state_dict.keys()):
    if key.startswith('head.'):
        del state_dict[key]
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα
def extract_features_from_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(image)
    return features.cpu().numpy().flatten()

# Συνάρτηση για τον υπολογισμό της ομοιότητας
def calculate_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Φόρτωση των δεδομένων και υπολογισμός ομοιότητας
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
case_distances = []

for case in sorted(os.listdir(base_dir)):
    case_dir = os.path.join(base_dir, case)
    if os.path.isdir(case_dir):
        images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
        abnormal = extract_features_from_image(model, images[0])
        normal1 = extract_features_from_image(model, images[1])
        normal2 = extract_features_from_image(model, images[2])
        sim1 = calculate_similarity(abnormal, normal1)
        sim2 = calculate_similarity(abnormal, normal2)
        avg_sim = (sim1 + sim2) / 2
        case_distances.append(1-avg_sim)
        print(f'Case {case}: Similarity with normal 1 = {sim1:.4f}, Similarity with normal 2 = {sim2:.4f}, Average similarity = {avg_sim:.4f}')

# Φόρτωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_LPIPS_VIT_SCORES.xlsx"
df = pd.read_excel(file_path)

# Ενημέρωση της στήλης ViT με τις νέες τιμές και αλλαγή ονόματος σε trainedViT
df = df.rename(columns={"ViT": "trainedViT"})
df['trainedViT'] = case_distances

# Αποθήκευση του ενημερωμένου αρχείου Excel
df.to_excel(file_path, index=False)
print("Excel file updated successfully.")