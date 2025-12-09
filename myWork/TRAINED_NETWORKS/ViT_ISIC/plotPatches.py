import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ορισμός των μετασχηματισμών
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT απαιτεί σταθερό μέγεθος εισόδου
    transforms.ToTensor()
])

# Φόρτωση και μετασχηματισμός της εικόνας
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

# Διαχωρισμός της εικόνας σε patches
def split_into_patches(image_tensor, patch_size=16):
    _, H, W = image_tensor.shape
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
    return patches

# Παράδειγμα χρήσης
image_path = r"C:\Users\steli\DIPLOMA\images\yeezyCARBON.png"  # Αντικαταστήστε με το πραγματικό μονοπάτι της εικόνας σας
image_tensor = load_and_transform_image(image_path)
patches = split_into_patches(image_tensor)

print(f"Number of patches: {patches.shape[0]}")
print(f"Patch shape: {patches.shape[1:]}")

# Τυπώστε τις τιμές των pixels για το πρώτο patch
first_patch = patches[0]
print("First patch pixel values:")
print(first_patch)

# Αν θέλετε να απεικονίσετε τα patches για να επιβεβαιώσετε τη διαδικασία
def plot_patches(patches):
    fig, axes = plt.subplots(14, 14, figsize=(10, 10))
    for i, patch in enumerate(patches):
        ax = axes[i // 14, i % 14]
        patch = patch.permute(1, 2, 0).numpy()
        patch = np.clip(patch, 0, 1)
        ax.imshow(patch)
        ax.axis('off')
    plt.show()

plot_patches(patches)
