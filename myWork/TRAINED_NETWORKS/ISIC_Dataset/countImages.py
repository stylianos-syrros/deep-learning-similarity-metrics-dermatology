import os

def count_images_in_directory(directory_path):
    subfolder_count = 0
    total_images_count = 0
    subfolder_image_counts = {}

    for subdir, dirs, files in os.walk(directory_path):
        if subdir == directory_path:
            subfolder_count = len(dirs)
        image_count = len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        if subdir != directory_path:
            subfolder_image_counts[subdir] = image_count
        total_images_count += image_count

    return subfolder_count, subfolder_image_counts, total_images_count

def print_directory_summary(directory_path):
    subfolder_count, subfolder_image_counts, total_images_count = count_images_in_directory(directory_path)
    
    print(f'Directory: {directory_path}')
    print(f'Number of subfolders: {subfolder_count}')
    for subfolder, image_count in subfolder_image_counts.items():
        print(f'{subfolder}: {image_count} images')
    print(f'Total number of images: {total_images_count}')

# Define the directories
test_directory = r"D:\Diploma\datasets\ISISC_Dataset\ISIC_2019_Test_Input"
train_directory = r"D:\Diploma\datasets\ISISC_Dataset\ISIC_2019_Training_Input"

# Print summary for each directory
#print("Test Directory Summary:")
#print_directory_summary(test_directory)
#print("\nTrain Directory Summary:")
#print_directory_summary(train_directory)

import os

# Λίστα με τα paths των φακέλων
folders = [
    r"D:\Diploma\datasets\processed_ISIC_Dataset\train\BCC",
    r"D:\Diploma\datasets\processed_ISIC_Dataset\train\MEL",
    r"D:\Diploma\datasets\processed_ISIC_Dataset\train\NV",
    r"D:\Diploma\datasets\processed_ISIC_Dataset\test\BCC",
    r"D:\Diploma\datasets\processed_ISIC_Dataset\test\MEL",
    r"D:\Diploma\datasets\processed_ISIC_Dataset\test\NV"
]

# Συνάρτηση για τον υπολογισμό του πλήθους των εικόνων σε κάθε φάκελο
def count_images(folder):
    return len([file for file in os.listdir(folder) if file.endswith('.jpg')])

# Υπολογισμός και εκτύπωση του πλήθους των εικόνων για κάθε φάκελο
for folder in folders:
    num_images = count_images(folder)
    print(f"Folder {folder} contains {num_images} images")

