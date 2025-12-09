import os
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from averageCosineSimilarity import calculate_cosine_similarity as avg_cosine_similarity
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

# Φόρτωση του εκπαιδευμένου μοντέλου VGG16
model_path = r"D:\Diploma\VGG16_models_ISIC\best_vgg16_model.pth"
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)  # Αλλαγή του τελευταίου fully connected layer για 3 κατηγορίες
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.to(device)
model.eval()

# Επιλογή των επιπέδων από τα οποία θα πάρεις τα χαρακτηριστικά
layer_name1 = 'features.28'  # (28): Conv2d
layer_name2 = 'features.29'  # (29): ReLU
layer_name3 = 'avgpool'      # (avgpool): AdaptiveAvgPool2d
layer_name4 = 'classifier.4' # (4): ReLU in classifier
layer_name5 = 'classifier.6' # (6): Linear(fc3)

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

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0).to(device)  # Προσθέτουμε την batch dimension και μεταφέρουμε στη συσκευή

# Διαδρομή προς τον κατάλογο με τα cases
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
case_distances_layer1_avg = []
case_distances_layer2_avg = []
case_distances_layer3_avg = []
case_distances_layer1_flat = []
case_distances_layer2_flat = []
case_distances_layer3_flat = []
case_distances_layer4_flat = []
case_distances_layer5_flat = []
cases = []

# Υπολογισμός απόστασης για κάθε case
for case in sorted(os.listdir(base_dir)):
    case_dir = os.path.join(base_dir, case)
    if os.path.isdir(case_dir):
        images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
        
        # Χαρακτηριστικά για το πρώτο layer
        img_tensor1 = preprocess_image(images[0])
        img_tensor2 = preprocess_image(images[1])
        img_tensor3 = preprocess_image(images[2])
        
        abnormal_layer1 = get_features_from_layer(model, img_tensor1, layer_name1)
        normal1_layer1 = get_features_from_layer(model, img_tensor2, layer_name1)
        normal2_layer1 = get_features_from_layer(model, img_tensor3, layer_name1)
        
        # Υπολογισμός της cosine similarity για το πρώτο layer χρησιμοποιώντας το averageCosineSimilarity
        sim1_layer1 = avg_cosine_similarity(abnormal_layer1, normal1_layer1)
        sim2_layer1 = avg_cosine_similarity(abnormal_layer1, normal2_layer1)
        avg_dist_layer1 = (1 - sim1_layer1 + 1 - sim2_layer1) / 2
        case_distances_layer1_avg.append(avg_dist_layer1)

        # Υπολογισμός της cosine similarity για το πρώτο layer χρησιμοποιώντας το flattenCosineSimilarity
        sim1_layer1 = flat_cosine_similarity(abnormal_layer1, normal1_layer1)
        sim2_layer1 = flat_cosine_similarity(abnormal_layer1, normal2_layer1)
        flat_dist_layer1 = (1 - sim1_layer1 + 1 - sim2_layer1) / 2
        case_distances_layer1_flat.append(flat_dist_layer1)

        # Χαρακτηριστικά για το δεύτερο layer
        abnormal_layer2 = get_features_from_layer(model, img_tensor1, layer_name2)
        normal1_layer2 = get_features_from_layer(model, img_tensor2, layer_name2)
        normal2_layer2 = get_features_from_layer(model, img_tensor3, layer_name2)

        # Υπολογισμός της cosine similarity για το δεύτερο layer χρησιμοποιώντας το averageCosineSimilarity
        sim1_layer2 = avg_cosine_similarity(abnormal_layer2, normal1_layer2)
        sim2_layer2 = avg_cosine_similarity(abnormal_layer2, normal2_layer2)
        avg_dist_layer2 = (1 - sim1_layer2 + 1 - sim2_layer2) / 2
        case_distances_layer2_avg.append(avg_dist_layer2)

        # Υπολογισμός της cosine similarity για το δεύτερο layer χρησιμοποιώντας το flattenCosineSimilarity
        sim1_layer2 = flat_cosine_similarity(abnormal_layer2, normal1_layer2)
        sim2_layer2 = flat_cosine_similarity(abnormal_layer2, normal2_layer2)
        flat_dist_layer2 = (1 - sim1_layer2 + 1 - sim2_layer2) / 2
        case_distances_layer2_flat.append(flat_dist_layer2)

        # Χαρακτηριστικά για το τρίτο layer
        abnormal_layer3 = get_features_from_layer(model, img_tensor1, layer_name3)
        normal1_layer3 = get_features_from_layer(model, img_tensor2, layer_name3)
        normal2_layer3 = get_features_from_layer(model, img_tensor3, layer_name3)

        # Υπολογισμός της cosine similarity για το τρίτο layer χρησιμοποιώντας το averageCosineSimilarity
        sim1_layer3 = avg_cosine_similarity(abnormal_layer3, normal1_layer3)
        sim2_layer3 = avg_cosine_similarity(abnormal_layer3, normal2_layer3)
        avg_dist_layer3 = (1 - sim1_layer3 + 1 - sim2_layer3) / 2
        case_distances_layer3_avg.append(avg_dist_layer3)

        # Υπολογισμός της cosine similarity για το τρίτο layer χρησιμοποιώντας το flattenCosineSimilarity
        sim1_layer3 = flat_cosine_similarity(abnormal_layer3, normal1_layer3)
        sim2_layer3 = flat_cosine_similarity(abnormal_layer3, normal2_layer3)
        flat_dist_layer3 = (1 - sim1_layer3 + 1 - sim2_layer3) / 2
        case_distances_layer3_flat.append(flat_dist_layer3)

        # Χαρακτηριστικά για το τέταρτο layer
        abnormal_layer4 = get_features_from_layer(model, img_tensor1, layer_name4)
        normal1_layer4 = get_features_from_layer(model, img_tensor2, layer_name4)
        normal2_layer4 = get_features_from_layer(model, img_tensor3, layer_name4)

        # Υπολογισμός της cosine similarity για το τέταρτο layer χρησιμοποιώντας το cosineSimilarity
        sim1_layer4 = cosine_similarity(abnormal_layer4, normal1_layer4)
        sim2_layer4 = cosine_similarity(abnormal_layer4, normal2_layer4)
        flat_dist_layer4 = (1 - sim1_layer4 + 1 - sim2_layer4) / 2
        case_distances_layer4_flat.append(flat_dist_layer4)

        # Χαρακτηριστικά για το τέταρτο layer
        abnormal_layer5 = get_features_from_layer(model, img_tensor1, layer_name5)
        normal1_layer5 = get_features_from_layer(model, img_tensor2, layer_name5)
        normal2_layer5 = get_features_from_layer(model, img_tensor3, layer_name5)

        # Υπολογισμός της cosine similarity για το τέταρτο layer χρησιμοποιώντας το cosineSimilarity
        sim1_layer5 = cosine_similarity(abnormal_layer5, normal1_layer5)
        sim2_layer5 = cosine_similarity(abnormal_layer5, normal2_layer5)
        flat_dist_layer5 = (1 - sim1_layer5 + 1 - sim2_layer5) / 2
        case_distances_layer5_flat.append(flat_dist_layer5)

        cases.append(case)
        print(f'Case {case}')

# Αποθήκευση των αποτελεσμάτων σε αρχείο Excel
file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances_Trained.xlsx"
df = pd.DataFrame({
    'Folder': cases,
    '(trained_avg)block5_conv3': case_distances_layer1_avg,
    '(trained_avg)block5_relu3': case_distances_layer2_avg,
    '(trained_avg)avgpool': case_distances_layer3_avg,
    '(trained_flat)block5_conv3': case_distances_layer1_flat,
    '(trained_flat)block5_relu3': case_distances_layer2_flat,
    '(trained_flat)avgpool': case_distances_layer3_flat,
    '(trained_flat)ReLU': case_distances_layer4_flat,
    '(trained_flat)fc3': case_distances_layer5_flat,
})

df.to_excel(file_path, index=False)
print("Excel file updated successfully.")