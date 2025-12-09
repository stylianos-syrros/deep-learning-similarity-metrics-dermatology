import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTModel, ViTImageProcessor

# Παράκαμψη της προειδοποίησης για symlinks
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Καταστολή συγκεκριμένης προειδοποίησης
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

# Ελέγχει αν η GPU είναι διαθέσιμη και θέτει την συσκευή
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Ορισμός των μετασχηματισμών με τα mean και std των εικόνων σου
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT απαιτεί σταθερό μέγεθος εισόδου
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.65886277, 0.45850426, 0.41027424], std=[0.14796351, 0.14156736, 0.13876683])
])

# Φόρτωση του εκπαιδευμένου ViT μοντέλου
trained_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # num_classes=0 to get features
best_model_path = r"D:\Diploma\models\best_model.pth"
state_dict = torch.load(best_model_path)
for key in list(state_dict.keys()):
    if key.startswith('head.'):
        del state_dict[key]
trained_model.load_state_dict(state_dict, strict=False)
trained_model.eval()
trained_model.to(device)

# Φόρτωση του προεκπαιδευμένου ViT μοντέλου
pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
pretrained_model.eval()
pretrained_model.to(device)

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα
def extract_features_from_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(image)
    return features.cpu().numpy().flatten()

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα με το προεκπαιδευμένο μοντέλο
def extract_features_from_image_pretrained(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    features = outputs.last_hidden_state  # Extract features from all patches including CLS token
    return features.cpu().numpy().flatten()

# Συνάρτηση για τον υπολογισμό της ομοιότητας
def calculate_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Φόρτωση των δεδομένων και υπολογισμός ομοιότητας
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
case_distances_trained = []
case_distances_pretrained = []
cases = []

for case in sorted(os.listdir(base_dir)):
    case_dir = os.path.join(base_dir, case)
    if os.path.isdir(case_dir):
        images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
        abnormal_trained = extract_features_from_image(trained_model, images[0], transform)
        normal1_trained = extract_features_from_image(trained_model, images[1], transform)
        normal2_trained = extract_features_from_image(trained_model, images[2], transform)
        sim1_trained = calculate_similarity(abnormal_trained, normal1_trained)
        sim2_trained = calculate_similarity(abnormal_trained, normal2_trained)
        avg_sim_trained = (sim1_trained + sim2_trained) / 2
        case_distances_trained.append(1 - avg_sim_trained)
        print(f'Case {case} (Trained ViT): Distance with normal 1 = {1-sim1_trained:.4f}, Distance with normal 2 = {1-sim2_trained:.4f}, Average distance = {1-avg_sim_trained:.4f}')

        abnormal_pretrained = extract_features_from_image_pretrained(pretrained_model, images[0], transform)
        normal1_pretrained = extract_features_from_image_pretrained(pretrained_model, images[1], transform)
        normal2_pretrained = extract_features_from_image_pretrained(pretrained_model, images[2], transform)
        sim1_pretrained = calculate_similarity(abnormal_pretrained, normal1_pretrained)  # Use only CLS token for similarity
        sim2_pretrained = calculate_similarity(abnormal_pretrained, normal2_pretrained)  # Use only CLS token for similarity
        avg_sim_pretrained = (sim1_pretrained + sim2_pretrained) / 2
        case_distances_pretrained.append(1 - avg_sim_pretrained)
        cases.append(case)
        print(f'Case {case} (Pretrained ViT): Distance with normal 1 = {1-sim1_pretrained:.4f}, Distance with normal 2 = {1-sim2_pretrained:.4f}, Average distance = {1-avg_sim_pretrained:.4f}')

# Εκτύπωση μηκών λιστών
print(f'Length of case_distances_trained: {len(case_distances_trained)}')
print(f'Length of case_distances_pretrained: {len(case_distances_pretrained)}')
print(f'Length of cases: {len(cases)}')

# Αποθήκευση των αποτελεσμάτων σε αρχείο Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ViT_Comparison_CLS.xlsx"
df = pd.DataFrame({
    'Folder': cases,
    'trainedViT': case_distances_trained,
    'pretrainedViT': case_distances_pretrained
})

df.to_excel(file_path, index=False)
print("Excel file updated successfully.")
