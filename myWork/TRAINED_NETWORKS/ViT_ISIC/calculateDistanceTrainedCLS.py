import os
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Έλεγχος για GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", "GPU" if torch.cuda.is_available() else "CPU")

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

# Φόρτωση του ViT μοντέλου
model_path = r"D:\Diploma\ViT_models_ISIC\best_vit_model.pth"
trained_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
checkpoint = torch.load(model_path)

# Φόρτωση των παραμέτρων στο μοντέλο
trained_model.load_state_dict(checkpoint['model_state_dict'])
#trained_model.load_state_dict(checkpoint, strict=False)
trained_model.eval()
trained_model.to(device)

# Φόρτωση και μετασχηματισμός της εικόνας
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Εξαγωγή του CLS token embedding
def get_cls_embedding(model, image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    cls_embedding = features[:, 0, :]  # Το πρώτο token είναι το CLS token
    return cls_embedding.cpu().numpy()

# Υπολογισμός της ομοιότητας
def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

# Υπολογισμός της μέσης απόστασης
def calculate_average_distance(case_folder):
    image_paths = [os.path.join(case_folder, f"{i}.jpg") for i in range(3)]
    cls_embeddings = [get_cls_embedding(trained_model, load_and_transform_image(image_path)) for image_path in image_paths]
    
    similarity_01 = calculate_similarity(cls_embeddings[0], cls_embeddings[1])
    similarity_02 = calculate_similarity(cls_embeddings[0], cls_embeddings[2])
    
    distance_01 = 1 - similarity_01
    distance_02 = 1 - similarity_02
    
    average_distance = (distance_01 + distance_02) / 2
    
    return distance_01, distance_02, average_distance

# Διατρέχουμε όλους τους φακέλους και υπολογίζουμε τις αποστάσεις
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
average_distances = []

for i in range(1, 177):
    case_folder = os.path.join(base_dir, f"CASE{i:03d}")
    dist_01, dist_02, avg_distance = calculate_average_distance(case_folder)
    average_distances.append((f"CASE{i:03d}", dist_01, dist_02, avg_distance))
    print(f"CASE{i:03d}: Dist_01 = {dist_01:.4f}, Dist_02 = {dist_02:.4f}, Average Distance = {avg_distance:.4f}")

# Ενημέρωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ViT_Comparison_Distances.xlsx"
df = pd.read_excel(file_path)

# Προσθήκη/ενημέρωση στήλης trainedViTCLS
df['trainedViTCLS'] = [dist[3] for dist in average_distances]

# Αποθήκευση του ενημερωμένου αρχείου Excel
df.to_excel(file_path, index=False)
print("Excel file updated successfully.")
