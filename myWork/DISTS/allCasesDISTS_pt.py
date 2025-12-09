import torch
import warnings
import torchvision.transforms as transforms
from PIL import Image
from DISTS_pytorch import DISTS
import os

warnings.filterwarnings("ignore")

# Δημιουργία ενός αντικειμένου DISTS
D = DISTS()

# Διαδρομή προς τον κατάλογο με τις εικόνες
base_dir = r"C:\Users\steli\DIPLOMA\bcc"

# Συνάρτηση για την προετοιμασία της εικόνας
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transforms.functional.resize(image, (224, 224))
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

# Υπολογισμός DISTS για όλους τους φακέλους
for folder in os.listdir(base_dir):
    case_path = os.path.join(base_dir, folder)
    if not os.path.isdir(case_path):
        continue
    
    # Φορτώνει τις εικόνες και υπολογίζει την απόσταση για τον πρώτο φάκελο
    image_paths = [os.path.join(case_path, f"{i}.jpg") for i in range(3)]
    image_tensors_DISTS = []
    for img_path in image_paths:
        image_tensor = prepare_image(img_path)
        image_tensors_DISTS.append(image_tensor)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = DISTS().to(device)
    #image_tensors_DISTS = [img_tensor.to(device) for img_tensor in image_tensors_DISTS]

    # Υπολογισμός DISTS για το ζεύγος (0,1) και (0,2)
    dists_01 = D(image_tensors_DISTS[0], image_tensors_DISTS[1]).item()
    dists_02 = D(image_tensors_DISTS[0], image_tensors_DISTS[2]).item()
    
    print(f"Folder: {folder}, DISTS (0,1): {dists_01}, DISTS (0,2): {dists_02}")
