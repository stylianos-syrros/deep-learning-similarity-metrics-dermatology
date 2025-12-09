import os
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ορισμός του μετασχηματισμού χωρίς κανονικοποίηση
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Κρατάμε την ίδια διάσταση με το δίκτυο
    transforms.ToTensor()
])

# Διαδρομή προς τον κατάλογο με τις εικόνες
base_dir = r"C:\Users\steli\DIPLOMA\bcc"

# Λίστα για αποθήκευση των τιμών των εικόνων
all_images = []

# Φόρτωση των εικόνων και αποθήκευση των τιμών τους
for folder in tqdm(os.listdir(base_dir)):
    case_path = os.path.join(base_dir, folder)
    if not os.path.isdir(case_path):
        continue
    for i in range(3):
        image_path = os.path.join(case_path, f"{i}.jpg")
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        all_images.append(image_tensor.numpy())

# Μετατροπή της λίστας σε numpy array
all_images = np.stack(all_images)

# Υπολογισμός των mean και std
mean = np.mean(all_images, axis=(0, 2, 3))
std = np.std(all_images, axis=(0, 2, 3))

print(f"Mean: {mean}")
print(f"Std: {std}")
