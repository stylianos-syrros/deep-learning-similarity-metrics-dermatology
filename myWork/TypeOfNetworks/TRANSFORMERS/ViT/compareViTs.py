import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTModel

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
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

# Φόρτωση του εκπαιδευμένου ViT μοντέλου
trained_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # num_classes=0 to get features
best_model_path = r"D:\Diploma\ViT_models_ISIC\best_vit_model.pth"
# Load the model state dict, ignoring the unexpected keys (head weights)
checkpoint = torch.load(best_model_path)

# Remove classification head weights if they are present in the checkpoint
filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not (k.startswith('head.weight') or k.startswith('head.bias'))}

trained_model.load_state_dict(filtered_state_dict, strict=False)
trained_model.to(device)
trained_model.eval()

# Φόρτωση του προεκπαιδευμένου ViT μοντέλου
pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
pretrained_model.eval()
pretrained_model.to(device)

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα (πλήρες embedding) από το εκπαιδευμένο μοντέλο
def extract_full_embedding_trained(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(image)
    return features.cpu().numpy().flatten()

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα με το CLS token για το εκπαιδευμένο μοντέλο
def extract_cls_embedding_trained(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(image)
    cls_embedding = features[:, 0, :]  # Extract the CLS token
    return cls_embedding.cpu().numpy().flatten()

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα (πλήρες embedding) από το προεκπαιδευμένο μοντέλο
def extract_full_embedding_pretrained(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    full_embedding = outputs.last_hidden_state.flatten(start_dim=1)  # Flatten all tokens
    print(f"Pretrained Model Full Embedding Shape: {full_embedding.shape}")  # Print the shape of the full embedding    
    return full_embedding.cpu().numpy().flatten()

# Συνάρτηση για την εξαγωγή χαρακτηριστικών από μια εικόνα με το CLS token για το προεκπαιδευμένο μοντέλο
def extract_cls_embedding_pretrained(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token
    print(f"Pretrained Model CLS Token Shape: {cls_embedding.shape}")  # Print the shape of the CLS token    
    return cls_embedding.cpu().numpy().flatten()

# Συνάρτηση για τον υπολογισμό της ομοιότητας
def calculate_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Φόρτωση των δεδομένων και υπολογισμός αποστάσεων
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
trained_full_distances = []
trained_cls_distances = []
pretrained_full_distances = []
pretrained_cls_distances = []
cases = []

for case in sorted(os.listdir(base_dir)):
    case_dir = os.path.join(base_dir, case)
    if os.path.isdir(case_dir):
        images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
        
        # Υπολογισμός αποστάσεων για το εκπαιδευμένο μοντέλο (full embedding)
        #abnormal_trained_full = extract_full_embedding_trained(trained_model, images[0], transform)
        #normal1_trained_full = extract_full_embedding_trained(trained_model, images[1], transform)
        #normal2_trained_full = extract_full_embedding_trained(trained_model, images[2], transform)
        #sim1_trained_full = calculate_similarity(abnormal_trained_full, normal1_trained_full)
        #sim2_trained_full = calculate_similarity(abnormal_trained_full, normal2_trained_full)
        #avg_sim_trained_full = (sim1_trained_full + sim2_trained_full) / 2
        #avg_dist_trained_full = 1 - avg_sim_trained_full
        #trained_full_distances.append(avg_dist_trained_full)
        
        # Υπολογισμός αποστάσεων για το εκπαιδευμένο μοντέλο (CLS token)
        #abnormal_trained_cls = extract_cls_embedding_trained(trained_model, images[0], transform)
        #normal1_trained_cls = extract_cls_embedding_trained(trained_model, images[1], transform)
        #normal2_trained_cls = extract_cls_embedding_trained(trained_model, images[2], transform)
        #sim1_trained_cls = calculate_similarity(abnormal_trained_cls, normal1_trained_cls)
        #sim2_trained_cls = calculate_similarity(abnormal_trained_cls, normal2_trained_cls)
        #avg_sim_trained_cls = (sim1_trained_cls + sim2_trained_cls) / 2
        #avg_dist_trained_cls = 1 - avg_sim_trained_cls
        #trained_cls_distances.append(avg_dist_trained_cls)
        
        # Υπολογισμός αποστάσεων για το προεκπαιδευμένο μοντέλο (full embedding)
        abnormal_pretrained_full = extract_full_embedding_pretrained(pretrained_model, images[0], transform)
        normal1_pretrained_full = extract_full_embedding_pretrained(pretrained_model, images[1], transform)
        normal2_pretrained_full = extract_full_embedding_pretrained(pretrained_model, images[2], transform)
        sim1_pretrained_full = calculate_similarity(abnormal_pretrained_full, normal1_pretrained_full)
        sim2_pretrained_full = calculate_similarity(abnormal_pretrained_full, normal2_pretrained_full)
        avg_sim_pretrained_full = (sim1_pretrained_full + sim2_pretrained_full) / 2
        avg_dist_pretrained_full = 1 - avg_sim_pretrained_full
        pretrained_full_distances.append(avg_dist_pretrained_full)
        
        # Υπολογισμός αποστάσεων για το προεκπαιδευμένο μοντέλο (CLS token)
        abnormal_pretrained_cls = extract_cls_embedding_pretrained(pretrained_model, images[0], transform)
        normal1_pretrained_cls = extract_cls_embedding_pretrained(pretrained_model, images[1], transform)
        normal2_pretrained_cls = extract_cls_embedding_pretrained(pretrained_model, images[2], transform)
        sim1_pretrained_cls = calculate_similarity(abnormal_pretrained_cls, normal1_pretrained_cls)
        sim2_pretrained_cls = calculate_similarity(abnormal_pretrained_cls, normal2_pretrained_cls)
        avg_sim_pretrained_cls = (sim1_pretrained_cls + sim2_pretrained_cls) / 2
        avg_dist_pretrained_cls = 1 - avg_sim_pretrained_cls
        pretrained_cls_distances.append(avg_dist_pretrained_cls)
        
#        cases.append(case)
#        print(f'Case {case} (Trained ViT): Full Embedding Distance = {avg_dist_trained_full:.4f}, CLS Token Distance = {avg_dist_trained_cls:.4f}')
#        print(f'Case {case} (Pretrained ViT): Full Embedding Distance = {avg_dist_pretrained_full:.4f}, CLS Token Distance = {avg_dist_pretrained_cls:.4f}')

# Φόρτωση του αρχείου Excel
#file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ViT_Comparison_Distances.xlsx"
#df = pd.DataFrame({
#    'Folder': cases,
#    'trainedViTEmb': trained_full_distances,
#    'trainedViTCLS': trained_cls_distances,
#    'pretrainedViTEmb': pretrained_full_distances,
#    'pretrainedViTCLS': pretrained_cls_distances
#})

# Αποθήκευση του ενημερωμένου αρχείου Excel
#df.to_excel(file_path, index=False)
#print("Excel file updated successfully.")
