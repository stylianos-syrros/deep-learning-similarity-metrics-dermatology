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

# Φόρτωση του ViT μοντέλου χωρίς κεφαλίδες (head layers)
model_path = r"D:\Diploma\ViT_models_ISIC\best_vit_model.pth"
trained_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # num_classes=0 to get features only
checkpoint = torch.load(model_path)
# Φόρτωση των παραμέτρων στο μοντέλο
trained_model.load_state_dict(checkpoint, strict=False)
trained_model.eval()
trained_model.to(device)

# Φόρτωση και μετασχηματισμός της εικόνας
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Εξαγωγή του CLS token embedding
def get_cls_embedding_trained(model, image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    cls_embedding = features[:, 0, :]  # Το πρώτο token είναι το CLS token
    return cls_embedding.cpu().numpy()

# Εξαγωγή ολόκληρου του embedding από το τελευταίο layer
def get_full_embedding_trained(model, image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    return features.cpu().numpy().flatten()

# Υπολογισμός της ομοιότητας
def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

# Υπολογισμός της μέσης απόστασης για το CLS token
def calculate_average_distance_cls(case_folder):
    image_paths = [os.path.join(case_folder, f"{i}.jpg") for i in range(3)]
    cls_embeddings = [get_cls_embedding_trained(trained_model, load_and_transform_image(image_path)) for image_path in image_paths]
    
    similarity_01 = calculate_similarity(cls_embeddings[0], cls_embeddings[1])
    similarity_02 = calculate_similarity(cls_embeddings[0], cls_embeddings[2])
    
    distance_01 = 1 - similarity_01
    distance_02 = 1 - similarity_02
    
    average_distance = (distance_01 + distance_02) / 2
    
    return distance_01, distance_02, average_distance

# Υπολογισμός της μέσης απόστασης για το πλήρες embedding
def calculate_average_distance_full(case_folder):
    image_paths = [os.path.join(case_folder, f"{i}.jpg") for i in range(3)]
    full_embeddings = [get_full_embedding_trained(trained_model, load_and_transform_image(image_path)) for image_path in image_paths]
    
    similarity_01 = calculate_similarity(full_embeddings[0], full_embeddings[1])
    similarity_02 = calculate_similarity(full_embeddings[0], full_embeddings[2])
    
    distance_01 = 1 - similarity_01
    distance_02 = 1 - similarity_02
    
    average_distance = (distance_01 + distance_02) / 2
    
    return distance_01, distance_02, average_distance

# Διατρέχουμε όλους τους φακέλους και υπολογίζουμε τις αποστάσεις
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
trained_full_distances = []
trained_cls_distances = []
cases = []

for i in range(1, 177):
    case_folder = os.path.join(base_dir, f"CASE{i:03d}")
    
    dist_01_cls, dist_02_cls, avg_distance_cls = calculate_average_distance_cls(case_folder)
    dist_01_full, dist_02_full, avg_distance_full = calculate_average_distance_full(case_folder)
    
    trained_cls_distances.append((f"CASE{i:03d}", dist_01_cls, dist_02_cls, avg_distance_cls))
    trained_full_distances.append((f"CASE{i:03d}", dist_01_full, dist_02_full, avg_distance_full))
    
    cases.append(f"CASE{i:03d}")
    
    print(f"CASE{i:03d}: CLS Token - Dist_01 = {dist_01_cls:.4f}, Dist_02 = {dist_02_cls:.4f}, Average Distance = {avg_distance_cls:.4f}")
    print(f"CASE{i:03d}: Full Embedding - Dist_01 = {dist_01_full:.4f}, Dist_02 = {dist_02_full:.4f}, Average Distance = {avg_distance_full:.4f}")

# Ενημέρωση του αρχείου Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Testing.xlsx"
if not os.path.exists(file_path):
    df = pd.DataFrame(columns=['Folder', 'trainedViTEmb', 'trainedViTCLS'])
else:
    df = pd.read_excel(file_path)

# Προσθήκη/ενημέρωση στήλης trainedViTEmb και trainedViTCLS
df['folder'] = cases
df['trainedViTEmb'] = [dist[3] for dist in trained_full_distances]
df['trainedViTCLS'] = [dist[3] for dist in trained_cls_distances]

# Αποθήκευση του ενημερωμένου αρχείου Excel
df.to_excel(file_path, index=False)
print("Excel file updated successfully.")
