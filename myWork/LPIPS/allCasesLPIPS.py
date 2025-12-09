import lpips
import torch
import warnings
import torchvision.transforms.functional as TF
from PIL import Image
import os

warnings.filterwarnings("ignore")

# Φόρτωση του μοντέλου LPIPS με το VGG
loss_fn_vgg = lpips.LPIPS(net='vgg')

# Διαδρομή προς τον κατάλογο με τις εικόνες
base_dir = r"C:\Users\steli\DIPLOMA\bcc"

# Υπολογισμός της LPIPS για όλους τους φακέλους
for folder in os.listdir(base_dir):
    case_path = os.path.join(base_dir, folder)
    if not os.path.isdir(case_path):
        continue
    
    # Φόρτωση των εικόνων
    image_paths = [os.path.join(case_path, f"{i}.jpg") for i in range(3)]
    image_tensors_LPIPS= []
    for img_path in image_paths:
        tempImageLPIPS = lpips.im2tensor(lpips.load_image(img_path))
        tempImageLPIPS = TF.resize(tempImageLPIPS, (224, 224))
        image_tensors_LPIPS.append(tempImageLPIPS)
    
    # Υπολογισμός της LPIPS για το ζεύγος (0,1) και (0,2)
    lpips_01 = loss_fn_vgg(image_tensors_LPIPS[0], image_tensors_LPIPS[1]).item()
    lpips_02 = loss_fn_vgg(image_tensors_LPIPS[0], image_tensors_LPIPS[2]).item()
    
    print(f"Folder: {folder}, LPIPS (0,1): {lpips_01}, LPIPS (0,2): {lpips_02}")
