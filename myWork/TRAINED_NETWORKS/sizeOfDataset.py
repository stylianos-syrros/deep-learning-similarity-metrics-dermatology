import os

def count_images_in_directory(directory_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    total_images = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_images += 1
    return total_images

# Διαδρομές των φακέλων
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"
train_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"

# Υπολογισμός αριθμού εικόνων σε κάθε φάκελο
test_images_count = count_images_in_directory(test_dir)
train_images_count = count_images_in_directory(train_dir)

print(f'Number of images in Test directory: {test_images_count}')
print(f'Number of images in Train directory: {train_images_count}')
