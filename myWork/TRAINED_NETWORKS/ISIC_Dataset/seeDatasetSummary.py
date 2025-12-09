import os
from torchvision import datasets

# Διαδρομές των φακέλων
train_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"

# Φόρτωση των δεδομένων από τους φακέλους
train_dataset = datasets.ImageFolder(train_dir)
test_dataset = datasets.ImageFolder(test_dir)

# Συνάρτηση για την καταμέτρηση των εικόνων σε κάθε κατηγορία
def count_images_per_class(dataset):
    class_counts = {cls: 0 for cls in dataset.classes}
    for _, label in dataset:
        class_counts[dataset.classes[label]] += 1
    return class_counts

# Καταμέτρηση των εικόνων σε κάθε dataset
train_class_counts = count_images_per_class(train_dataset)
test_class_counts = count_images_per_class(test_dataset)

# Εκτύπωση των αποτελεσμάτων
print(f'Total number of images in training dataset: {len(train_dataset)}')
print(f'Total number of images in test dataset: {len(test_dataset)}')

print('\nNumber of images per class in training dataset:')
for class_name, count in train_class_counts.items():
    print(f'{class_name}: {count} images')

print('\nNumber of images per class in test dataset:')
for class_name, count in test_class_counts.items():
    print(f'{class_name}: {count} images')

# Εκτύπωση του πλήθους των κατηγοριών
print(f'\nNumber of classes: {len(train_dataset.classes)}')
