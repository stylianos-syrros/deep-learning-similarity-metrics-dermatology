import os
import zipfile
import torch  # Εισαγωγή της βιβλιοθήκης torch

# Ορισμός του path για την αποθήκευση των δεδομένων
dataset_dir = r"C:\Users\steli\DIPLOMA\myProgramms\TRAINED_NETWORKS\Skin_Cancer_Dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Λήψη του dataset από το Kaggle
os.system("kaggle datasets download -d nodoubttome/skin-cancer9-classesisic -p " + dataset_dir)

# Εξαγωγή του αρχείου ZIP
zip_path = os.path.join(dataset_dir, 'skin-cancer9-classesisic.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

print('Dataset downloaded and extracted!')

# Έλεγχος του περιεχομένου του directory
extracted_folders = os.listdir(dataset_dir)
print("Extracted folders:", extracted_folders)

# Ενημέρωση του path του image_dir ανάλογα με την δομή
# Ελέγχουμε αν τα αρχεία βρίσκονται απευθείας στο dataset_dir ή σε υποφακέλους
image_dir = dataset_dir
if 'skin-cancer9-classesisic' in extracted_folders:
    image_dir = os.path.join(dataset_dir, 'skin-cancer9-classesisic')
else:
    subfolders = [f for f in extracted_folders if os.path.isdir(os.path.join(dataset_dir, f))]
    if len(subfolders) == 1:
        image_dir = os.path.join(dataset_dir, subfolders[0])

print("Using image directory:", image_dir)

# Φόρτωση δεδομένων (ενδεικτικός κώδικας)
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Φόρτωση του dataset
dataset = datasets.ImageFolder(image_dir, transform=transform)

# Διαχωρισμός σε training και validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f'Training samples: {train_size}, Validation samples: {val_size}')
