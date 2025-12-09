import os
import pandas as pd

# Διαδρομές των φακέλων και του αρχείου CSV
train_folder = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"
test_folder = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"
csv_file = r"D:\Diploma\datasets\ISIC_2019_Training_GroundTruth.csv"

# Φόρτωση του CSV αρχείου
df = pd.read_csv(csv_file)

# Κατηγορίες για έλεγχο
categories = ['MEL', 'NV', 'BCC']

def get_image_files(folder):
    image_files = {}
    for category in categories:
        category_folder = os.path.join(folder, category)
        if os.path.exists(category_folder):
            image_files[category] = set(os.listdir(category_folder))
    return image_files

# Λήψη των εικόνων από τους φακέλους train και test
train_images = get_image_files(train_folder)
test_images = get_image_files(test_folder)

# Έλεγχος για κοινές εικόνες μεταξύ κατηγοριών στο train και στο test
def check_common_images(images_dict):
    common_images = {}
    for category1 in categories:
        for category2 in categories:
            if category1 != category2:
                common = images_dict[category1].intersection(images_dict[category2])
                if common:
                    common_images[f"{category1} and {category2}"] = common
    return common_images

train_common_images = check_common_images(train_images)
test_common_images = check_common_images(test_images)

# Έλεγχος αν οι εικόνες ανήκουν στις σωστές κατηγορίες σύμφωνα με το αρχείο CSV
def check_image_categories(images_dict, df):
    incorrect_images = {}
    for category in categories:
        incorrect_images[category] = []
        for img in images_dict[category]:
            img_id = img.split('.')[0]  # Αφαίρεση της επέκτασης αρχείου
            if df.loc[df['image'] == img_id, category].values[0] != 1:
                incorrect_images[category].append(img)
    return incorrect_images

train_incorrect_images = check_image_categories(train_images, df)
test_incorrect_images = check_image_categories(test_images, df)

# Εμφάνιση αποτελεσμάτων
print("Common images between categories in train:")
for key, value in train_common_images.items():
    print(f"{key}: {len(value)} images")

print("Common images between categories in test:")
for key, value in test_common_images.items():
    print(f"{key}: {len(value)} images")

print("\nIncorrect images in train:")
for category, images in train_incorrect_images.items():
    print(f"{category}: {len(images)} incorrect images")

print("\nIncorrect images in test:")
for category, images in test_incorrect_images.items():
    print(f"{category}: {len(images)} incorrect images")
