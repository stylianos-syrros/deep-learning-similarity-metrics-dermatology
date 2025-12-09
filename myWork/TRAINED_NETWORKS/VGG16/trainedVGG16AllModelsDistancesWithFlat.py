import os
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from flattenCosineSimilarity import calculate_cosine_similarity as flat_cosine_similarity
import warnings

# Ελέγχει αν η GPU είναι διαθέσιμη και θέτει την συσκευή
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Καταστολή προειδοποιήσεων
warnings.filterwarnings('ignore')

# Ορισμός των μετασχηματισμών με τα mean και std των εικόνων σας
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

# Φόρτωση του VGG16 μοντέλου
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)  # Αλλαγή του τελευταίου fully connected layer για 3 κατηγορίες
model.to(device)
model.eval()

# Διαδρομή προς τον κατάλογο με τα cases
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
cases = [case for case in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, case))]

# Λειτουργία για την προεπεξεργασία των εικόνων
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Λειτουργία για την εξαγωγή χαρακτηριστικών από συγκεκριμένα επίπεδα
def get_features_from_layer(model, image_tensor, layer_name):
    with torch.no_grad():
        if 'features' in layer_name:
            x = model.features[:int(layer_name.split('.')[1]) + 1](image_tensor)
        elif 'avgpool' in layer_name:
            x = model.features(image_tensor)
            x = model.avgpool(x)
        else:
            x = model.features(image_tensor)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier[:int(layer_name.split('.')[1]) + 1](x)
    return x.cpu().numpy()

# Λειτουργία για τον υπολογισμό της μέσης απόστασης χρησιμοποιώντας cosine similarity
def calculate_flat_distance(abnormal_layer, normal1_layer, normal2_layer):
    sim1 = flat_cosine_similarity(abnormal_layer, normal1_layer)
    sim2 = flat_cosine_similarity(abnormal_layer, normal2_layer)
    avg_flat_dist = (1 - sim1 + 1 - sim2) / 2
    return avg_flat_dist

# Επίπεδα από τα οποία θα πάρεις τα χαρακτηριστικά
layer_name1 = 'avgpool'  # (avgpool): AdaptiveAvgPool2d
layer_name2 = 'classifier.4'  # (4): ReLU in classifier

# Φάκελος αποθήκευσης μοντέλων
model_dir = r"D:\Diploma\VGG16_models_ISIC"

# Λίστα με τις εποχές που θα εξετάσουμε
epochs = range(1, 26)  # Εποχές από 1 μέχρι 25

# Δημιουργία DataFrame για την αποθήκευση των αποτελεσμάτων όλων των εποχών
all_results = []

# Προσθήκη κεφαλίδων για τις στήλες του DataFrame
columns = ['Case'] + [f'Epoch_{epoch}_avgpool' for epoch in epochs] + [f'Epoch_{epoch}_classifier.4' for epoch in epochs]

# Υπολογισμός αποστάσεων για κάθε επιλεγμένη εποχή
for epoch in epochs:
    model_path = os.path.join(model_dir, f'vgg16_model_epoch_{epoch}.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for case in cases:
        case_dir = os.path.join(base_dir, case)
        images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]

        img_tensor1 = preprocess_image(images[0])
        img_tensor2 = preprocess_image(images[1])
        img_tensor3 = preprocess_image(images[2])

        # Χαρακτηριστικά για το πρώτο layer
        abnormal_layer1 = get_features_from_layer(model, img_tensor1, layer_name1)
        normal1_layer1 = get_features_from_layer(model, img_tensor2, layer_name1)
        normal2_layer1 = get_features_from_layer(model, img_tensor3, layer_name1)
        avg_flat_dist1 = calculate_flat_distance(abnormal_layer1, normal1_layer1, normal2_layer1)

        # Χαρακτηριστικά για το δεύτερο layer
        abnormal_layer2 = get_features_from_layer(model, img_tensor1, layer_name2)
        normal1_layer2 = get_features_from_layer(model, img_tensor2, layer_name2)
        normal2_layer2 = get_features_from_layer(model, img_tensor3, layer_name2)
        avg_flat_dist2 = calculate_flat_distance(abnormal_layer2, normal1_layer2, normal2_layer2)

        # Βρες ή πρόσθεσε την εγγραφή για το case στον πίνακα αποτελεσμάτων
        result_entry = next((entry for entry in all_results if entry[0] == case), None)
        if result_entry is None:
            result_entry = [case] + [None] * (2 * len(epochs))
            all_results.append(result_entry)
        
        epoch_index = epochs.index(epoch)
        result_entry[1 + epoch_index] = avg_flat_dist1
        result_entry[1 + len(epochs) + epoch_index] = avg_flat_dist2


# Δημιουργία DataFrame και αποθήκευση σε Excel
df = pd.DataFrame(all_results, columns=columns)
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_VGG16_Distances_All_Epochs.xlsx"
df.to_excel(file_path, index=False)
print(f"Excel file with all epochs updated successfully.")

print("All epochs processed.")
