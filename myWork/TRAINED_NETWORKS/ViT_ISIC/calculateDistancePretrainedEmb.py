import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from transformers import ViTModel
from sklearn.metrics.pairwise import cosine_similarity

# Παράκαμψη της προειδοποίησης για symlinks
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

# Φόρτωση του προεκπαιδευμένου ViT μοντέλου
pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
pretrained_model.eval()
pretrained_model.to(device)

# Φόρτωση και μετασχηματισμός της εικόνας
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Εξαγωγή ολόκληρου του embedding από το τελευταίο layer
def get_full_embedding_pretrained(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    full_embedding = outputs.last_hidden_state.flatten(start_dim=1)  # Flatten all tokens
    return full_embedding.cpu().numpy().flatten()

# Υπολογισμός της ομοιότητας
def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

# Υπολογισμός της μέσης απόστασης
def calculate_average_distance(case_folder):
    image_paths = [os.path.join(case_folder, f"{i}.jpg") for i in range(3)]
    full_embeddings = [get_full_embedding_pretrained(pretrained_model, load_and_transform_image(image_path)) for image_path in image_paths]
    
    print("1: Shape of full_embeddings[0]:", full_embeddings[0].shape)
    print("1: Type of full_embeddings[0]:", type(full_embeddings[0]))

    similarity_01 = calculate_similarity(full_embeddings[0], full_embeddings[1])
    similarity_02 = calculate_similarity(full_embeddings[0], full_embeddings[2])
    
    distance_01 = 1 - similarity_01
    distance_02 = 1 - similarity_02
    
    average_distance = (distance_01 + distance_02) / 2
    
    return distance_01, distance_02, average_distance

# Διατρέχουμε όλους τους φακέλους και υπολογίζουμε τις αποστάσεις
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
pretrained_full_distances = []
cases = []

#for i in range(1, 177):
for i in range(1, 3):
    case_folder = os.path.join(base_dir, f"CASE{i:03d}")
    dist_01, dist_02, avg_distance = calculate_average_distance(case_folder)
    pretrained_full_distances.append((f"CASE{i:03d}", dist_01, dist_02, avg_distance))
    cases.append(f"CASE{i:03d}")
    print(f"CASE{i:03d}: Dist_01 = {dist_01:.4f}, Dist_02 = {dist_02:.4f}, Average Distance = {avg_distance:.4f}")

# Ενημέρωση του αρχείου Excel
#file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ViT_Comparison_Distances.xlsx"
#df = pd.read_excel(file_path)

# Προσθήκη/ενημέρωση στήλης pretrainedViTEmb
#df['pretrainedViTEmb'] = [dist[3] for dist in pretrained_full_distances]

# Αποθήκευση του ενημερωμένου αρχείου Excel
#df.to_excel(file_path, index=False)
#print("Excel file updated successfully.")
