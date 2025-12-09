import os
import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_image(image_path):
    return Image.open(image_path)

def extract_features(image_path, model, feature_extractor):
    image = load_image(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :]
    spatial_tokens = outputs.last_hidden_state[:, 1:, :]
    combined_vector = torch.cat((cls_token, spatial_tokens.view(spatial_tokens.size(0), -1)), dim=1)
    return combined_vector.detach().numpy()

def calculate_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)[0][0]

# Έλεγχος αν υπάρχει διαθέσιμη GPU και χρήση της
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Φόρτωση μοντέλου και feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Καθορισμός root φακέλου
root_folder = r"C:\Users\steli\DIPLOMA\bcc"

# Επεξεργασία όλων των cases
case_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
mean_distance = 0
total_cases = len(case_folders)

for case_folder in case_folders:
    case_name = os.path.basename(case_folder)
    image_paths = [os.path.join(case_folder, f"{i}.jpg") for i in range(3)]
    vectors = [extract_features(image_path, model, feature_extractor) for image_path in image_paths]

    dist_01 = calculate_similarity(vectors[0], vectors[1])
    dist_02 = calculate_similarity(vectors[0], vectors[2])
    mean_dist = (dist_01+dist_01)/2

    print(f"Case: {case_name} | Distance between 0 and 1: {dist_01} | Distance between 0 and 2: {dist_02} | Mean Distance: {mean_dist}")
    

