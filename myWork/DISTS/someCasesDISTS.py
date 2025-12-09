import torch
import warnings
import torchvision.transforms as transforms
from PIL import Image
from DISTS_pytorch import DISTS
import os
import time

warnings.filterwarnings("ignore")

# Δημιουργία ενός αντικειμένου DISTS
D = DISTS()

# Διαδρομή προς τον κατάλογο με τις εικόνες
base_dir = r"C:\Users\steli\DIPLOMA\bcc"

# Υπολογισμός DISTS για τους πρώτους πέντε φακέλους
start_time = time.time()  # Έναρξη χρονομέτρησης

for i in range(1, 6):
    case_dir = f"CASE{i:03d}"  # Δημιουργία ονόματος φακέλου με μηδικά μηδενικά
    case_path = os.path.join(base_dir, case_dir)
    
    # Εξαγωγή των ονομάτων αρχείων φωτογραφιών
    photo_filenames = [f"{i}.jpg" for i in range(3)]
    
    # Εναλλαγή μεταξύ των δύο πρώτων φωτογραφιών
    for i in range(len(photo_filenames) - 1):
        image1_path = os.path.join(case_path, photo_filenames[0])
        image2_path = os.path.join(case_path, photo_filenames[i+1])

        # Φόρτωση των εικόνων σε μορφή PIL
        image1_pil = Image.open(image1_path).convert("RGB")
        image2_pil = Image.open(image2_path).convert("RGB")

        # Ορισμός των ίδιων διαστάσεων για τις εικόνες
        target_size = (224, 224)
        image1_pil = image1_pil.resize(target_size)
        image2_pil = image2_pil.resize(target_size)

        # Μετατροπή των εικόνων σε Tensor
        transform = transforms.ToTensor()
        image1_tensor = transform(image1_pil).unsqueeze(0)  # Προσθέστε τον διάσταση του batch
        image2_tensor = transform(image2_pil).unsqueeze(0)  # Προσθέστε τον διάσταση του batch

        # Υπολογισμός DISTS μεταξύ των δύο εικόνων
        dists_value = D(image1_tensor, image2_tensor)
        print(f"DISTS Value for {photo_filenames[0]} and {photo_filenames[i+1]} in {case_dir}: {dists_value.item()}")

end_time = time.time()  # Λήξη χρονομέτρησης
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")  # Εκτύπωση του χρόνου εκτέλεσης
