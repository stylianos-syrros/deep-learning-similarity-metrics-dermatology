import torch
import warnings
import torchvision.transforms as transforms
from PIL import Image
from DISTS_pytorch import DISTS

warnings.filterwarnings("ignore")

# Διαβάστε τις εικόνες
image1_path = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
image2_path = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"

# Φορτώστε τις εικόνες σε μορφή PIL
image1_pil = Image.open(image1_path).convert("RGB")
image2_pil = Image.open(image2_path).convert("RGB")

# Ορίστε τις ίδιες διαστάσεις για τις εικόνες
target_size = (224, 224)
image1_pil = image1_pil.resize(target_size)
image2_pil = image2_pil.resize(target_size)

# Ορίστε τις μετασχηματισμούς που απαιτούνται για το DISTS μοντέλο
transform = transforms.Compose([
    transforms.ToTensor(),   # Μετατροπή σε Tensor
])

# Εφαρμόστε τους μετασχηματισμούς στις εικόνες
image1_tensor = transform(image1_pil).unsqueeze(0)  # Προσθέστε τον διάσταση του batch
image2_tensor = transform(image2_pil).unsqueeze(0)  # Προσθέστε τον διάσταση του batch

print("Ο τύπος του region1_img_resized είναι:", type(image1_tensor))
print("Ο τύπος του region2_img_resized είναι:", type(image2_tensor))# Δημιουργία ενός αντικειμένου DISTS

D = DISTS()

# Υπολογισμός DISTS μεταξύ των δύο εικόνων
dists_value = D(image1_tensor, image2_tensor)
print("DISTS Value:", dists_value.item())

# Υπολογισμός της απώλειας DISTS (loss)
dists_loss = D(image1_tensor, image2_tensor, require_grad=True, batch_average=True)
dists_loss.backward()
print("DISTS Loss:", dists_loss.item())

#for name, param in D.named_parameters():
#    print(name, param.size())
