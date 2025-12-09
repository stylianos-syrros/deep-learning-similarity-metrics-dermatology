import os
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Ρύθμιση της συσκευής
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Απενεργοποίηση προειδοποιήσεων
warnings.filterwarnings('ignore')

# Μετασχηματισμοί για τις εικόνες
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6459899, 0.52058024, 0.51453681], std=[0.14522511, 0.15518147, 0.16543107])
])

# Φόρτωση του μοντέλου VGG16
model_path = r"D:\Diploma\VGG16_models_ISIC\vgg16_model_epoch_4.pth"
model_epoch_4 = models.vgg16(pretrained=False)
model_epoch_4.classifier[6] = torch.nn.Linear(model_epoch_4.classifier[6].in_features, 3)
model_epoch_4.load_state_dict(torch.load(model_path)['model_state_dict'])
model_epoch_4.to(device)
model_epoch_4.eval()

# Διαδρομή για τα cases
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
cases = [case for case in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, case))]

def get_features(model, img_path, layer_name):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img_tensor = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        if layer_name == 'features.28(block5_covn3)':
            features = model.features[:29](img_tensor)  # Παίρνει τα χαρακτηριστικά μέχρι το features.29
        elif layer_name == 'features.29(block5_relu3)':
            features = model.features[:30](img_tensor)  # Παίρνει τα χαρακτηριστικά μέχρι το features.30
        elif layer_name == 'features.30(block5_maxpool)':
            features = model.features[:31](img_tensor)  # Παίρνει τα χαρακτηριστικά μέχρι το features.31
        elif layer_name == 'allChannels':
            features = model.features(img_tensor)
            features = features[:, :512, :, :]  # Παίρνει όλα τα κανάλια(512) αφού η εικόνα έχει περάσει και από τα 31 επίπεδα
        elif layer_name == 'block5':
            features = model.features(img_tensor) # Παίρνει τα χαρακτηριστικά από όλο το block5
        elif layer_name == 'avgpool':
            features = model.features(img_tensor) # Παίρνει τα χαρακτηριστικά από το avgpool επίπεδο
            features = model.avgpool(features)
        else:
            raise ValueError("Invalid layer name specified.")
    
    return features.squeeze().view(features.shape[1], -1).cpu().numpy()

# Συνάρτηση για τον υπολογισμό της μέσης απόστασης
def calculate_mean_distance(normal_layer, abnormal1_layer, abnormal2_layer):
    sim1 = cosine_similarity(normal_layer.reshape(1, -1), abnormal1_layer.reshape(1, -1))[0][0]
    sim2 = cosine_similarity(normal_layer.reshape(1, -1), abnormal2_layer.reshape(1, -1))[0][0]
    mean_distance = (1 - sim1 + 1 - sim2) / 2
    return mean_distance

# Υπολογισμός της μέσης απόστασης για κάθε case και για κάθε layer
results = []
for case in cases:
    case_dir = os.path.join(base_dir, case)
    images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
    
    features_28 = [get_features(model_epoch_4, img, 'features.28(block5_covn3)') for img in images]
    features_29 = [get_features(model_epoch_4, img, 'features.29(block5_relu3)') for img in images]
    features_30 = [get_features(model_epoch_4, img, 'features.30(block5_maxpool)') for img in images]
    features_allChannels = [get_features(model_epoch_4, img, 'allChannels') for img in images]
    features_block5 = [get_features(model_epoch_4, img, 'block5') for img in images]
    features_avgpool = [get_features(model_epoch_4, img, 'avgpool') for img in images]


    dist_features_28 = calculate_mean_distance(features_28[0], features_28[1], features_28[2])
    dist_features_29 = calculate_mean_distance(features_29[0], features_29[1], features_29[2])
    dist_features_30 = calculate_mean_distance(features_30[0], features_30[1], features_30[2])
    dist_features_allChannels = calculate_mean_distance(features_allChannels[0], features_allChannels[1], features_allChannels[2])
    dist_features_block5 = calculate_mean_distance(features_block5[0], features_block5[1], features_block5[2])
    dist_features_avgpool = calculate_mean_distance(features_avgpool[0], features_avgpool[1], features_avgpool[2])


    results.append({
        'Case': case,
        'Distance features.28': dist_features_28,
        'Distance features.29': dist_features_29,
        'Distance features.30': dist_features_30,
        'Distance features.allChannels': dist_features_allChannels,
        'Distance features.block5': dist_features_block5,
        'Distance features.avgpool': dist_features_avgpool
    })

# Αποθήκευση σε Excel
df = pd.DataFrame(results)
df.to_excel(r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DifferentVGG16_layers.xlsx", index=False)
print("Results saved successfully.")
