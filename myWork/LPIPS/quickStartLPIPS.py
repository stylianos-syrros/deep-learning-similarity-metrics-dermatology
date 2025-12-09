import lpips
import torch
import warnings
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.utils import save_image


warnings.filterwarnings("ignore")

# Δημιουργία αντικειμένου LPIPS για το δίκτυο Alex
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

# Δημιουργία αντικειμένου LPIPS για το δίκτυο VGG
#loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

# Φορτώνουμε τις δύο εικόνες
img_path_0 = r"C:\Users\steli\DIPLOMA\bcc\CASE004\0.jpg"
img_path_1 = r"C:\Users\steli\DIPLOMA\bcc\CASE004\1.jpg"

# Φορτώνουμε τις εικόνες με χρήση του torch
img0 = lpips.im2tensor(lpips.load_image(img_path_0))
img1 = lpips.im2tensor(lpips.load_image(img_path_1))
img0 = TF.resize(img0, (224, 224))
img1 = TF.resize(img1, (224, 224))


# Υπολογισμός της μετρικής LPIPS για τις δύο εικόνες χρησιμοποιώντας το δίκτυο Alex
d_alex = loss_fn_alex(img0, img1)

print("Type of img0:", type(img0))
print("Type of img1:", type(img1))
print("Shape of img0:",img0.shape)
print("Shape of img1:",img1.shape)

# Υπολογισμός της μετρικής LPIPS για τις δύο εικόνες χρησιμοποιώντας το δίκτυο VGG
#d_vgg = loss_fn_vgg(img0, img1)

# Εκτύπωση της τιμής της μετρικής LPIPS για το δίκτυο ALEX
print("LPIPS value for ALEX:", d_alex.item())

# Εκτύπωση της τιμής της μετρικής LPIPS για το δίκτυο VGG
#print("LPIPS value for VGG:", d_vgg.item())