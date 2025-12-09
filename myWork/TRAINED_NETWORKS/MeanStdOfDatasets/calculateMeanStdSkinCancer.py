import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ορισμός των βασικών μετασχηματισμών χωρίς κανονικοποίηση
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Ορισμός της διαδρομής του νέου dataset
train_dir = r"C:\Users\steli\DIPLOMA\myProgramms\TRAINED_NETWORKS\Skin_Cancer_Dataset\SkinCancerISIC\Train"

# Έλεγχος αν η διαδρομή υπάρχει
if not os.path.exists(train_dir):
    print(f"Directory {train_dir} does not exist.")
else:
    print(f"Directory {train_dir} exists.")

# Φόρτωση του νέου dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Υπολογισμός των mean και std
mean = 0.0
std = 0.0
nb_samples = 0

for data, _ in train_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("Mean:", mean)
print("Std:", std)
