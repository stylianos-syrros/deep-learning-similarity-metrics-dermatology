import os
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Διαδρομή προς τον κατάλογο με τις εικόνες
dataset_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"

# Μετασχηματισμός εικόνας σε tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Αρχικοποίηση μεταβλητών
mean = np.zeros(3)
std = np.zeros(3)
n_images = 0

# Διαδρομή μέσα στους φακέλους και υπολογισμός mean και std
for root, _, files in os.walk(dataset_dir):
    for file in tqdm(files):
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image)
            mean += img_tensor.mean(dim=(1, 2)).numpy()
            std += img_tensor.std(dim=(1, 2)).numpy()
            n_images += 1

mean /= n_images
std /= n_images

print("Mean:", mean)
print("Std:", std)
