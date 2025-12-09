import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Διαδρομές των φακέλων
input_folder = r"D:\Diploma\datasets\ISISC_Dataset\ISIC_2019_Training_Input\ISIC_2019_Training_Input"
output_folder = r"D:\Diploma\datasets\processed_ISIC_Dataset"
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

# Διαγραφή των περιεχομένων των φακέλων train και test αν υπάρχουν
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

clear_folder(train_folder)
clear_folder(test_folder)

# Δημιουργία των φακέλων train και test αν δεν υπάρχουν
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Κατηγορίες για τις οποίες θέλουμε να διατηρήσουμε τις εικόνες
categories = ['MEL', 'NV', 'BCC']

# Διαδρομή του CSV αρχείου
csv_file = r"D:\Diploma\datasets\ISIC_2019_Training_GroundTruth.csv"

# Φόρτωση του CSV αρχείου
df = pd.read_csv(csv_file)

# Διατήρηση μόνο των κατηγοριών που μας ενδιαφέρουν
df = df[df[categories].sum(axis=1) > 0]

# Επιλογή περίπου 3100 εικόνων για κάθε κατηγορία
sampled_df = pd.concat([df[df[category] == 1].sample(3100, random_state=42) for category in categories])

# Χωρισμός των εικόνων σε train και test
train_df, test_df = train_test_split(sampled_df, test_size=0.2, random_state=42)

def copy_images(df, dest_folder):
    for idx, row in df.iterrows():
        for category in categories:
            if row[category] == 1:
                category_folder = os.path.join(dest_folder, category)
                os.makedirs(category_folder, exist_ok=True)
                img_name = row['image'] + '.jpg'
                src_path = os.path.join(input_folder, img_name)
                dst_path = os.path.join(category_folder, img_name)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    print(f"Image {img_name} not found in {input_folder}")

# Αντιγραφή των εικόνων στους φακέλους train και test
copy_images(train_df, train_folder)
copy_images(test_df, test_folder)

print("Dataset created successfully.")

# Εμφάνιση του αριθμού των εικόνων σε κάθε φάκελο
def count_images(folder):
    return len([file for file in os.listdir(folder) if file.endswith('.jpg')])

folders = [
    os.path.join(train_folder, 'BCC'),
    os.path.join(train_folder, 'MEL'),
    os.path.join(train_folder, 'NV'),
    os.path.join(test_folder, 'BCC'),
    os.path.join(test_folder, 'MEL'),
    os.path.join(test_folder, 'NV')
]

for folder in folders:
    num_images = count_images(folder)
    print(f"Folder {folder} contains {num_images} images")
