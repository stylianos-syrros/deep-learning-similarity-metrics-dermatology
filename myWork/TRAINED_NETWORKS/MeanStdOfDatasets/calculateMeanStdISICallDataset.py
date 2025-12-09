import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_mean_std(image_paths):
    """Υπολογίζει τις μέσες τιμές (mean) και τις τυπικές αποκλίσεις (std) για μια λίστα από εικόνες."""
    num_channels = 3  # Για εικόνες RGB
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)
    num_pixels = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img) / 255.0  # Κανονικοποίηση εικόνας
        channel_sum += np.sum(img_np, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img_np), axis=(0, 1))
        num_pixels += img_np.shape[0] * img_np.shape[1]

    mean = channel_sum / num_pixels
    std = np.sqrt(channel_sum_squared / num_pixels - np.square(mean))

    return mean, std

def get_image_paths(root_dir):
    """Επιστρέφει μια λίστα με όλα τα paths των εικόνων σε έναν ριζικό φάκελο."""
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

if __name__ == "__main__":
    train_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"
    test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"

    print("Gathering image paths...")
    train_image_paths = get_image_paths(train_dir)
    test_image_paths = get_image_paths(test_dir)

    all_image_paths = train_image_paths + test_image_paths

    print(f"Found {len(all_image_paths)} images.")

    mean, std = calculate_mean_std(all_image_paths)
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")
