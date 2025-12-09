import os
import warnings
import logging
import tensorflow as tf
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
from scipy.spatial import distance

# Καταστολή προειδοποιήσεων και μηνυμάτων
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')
tf.get_logger().setLevel(logging.ERROR)

# Φόρτωση του προεκπαιδευμένου ViT μοντέλου και του image processor
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Μέγεθος εικόνας που αναμένεται από το μοντέλο ViT
image_size = (224, 224)

def preprocess_image(image_path):
    # Άνοιγμα και προεπεξεργασία της εικόνας
    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

def get_deep_features(image_path):
    # Λήψη βαθιών χαρακτηριστικών από την εικόνα
    inputs = preprocess_image(image_path)
    outputs = model(**inputs)
    # Χρησιμοποιούμε την έξοδο από το CLS token
    features = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return features.flatten()

# Κεντρικός κατάλογος που περιέχει όλους τους φακέλους
base_folder = r"C:\Users\steli\DIPLOMA\bcc"

# Συνολικός αριθμός φακέλων
num_folders = 176

for i in range(1, num_folders + 1):
    folder = os.path.join(base_folder, f"CASE{i:03}")
    image_path0 = os.path.join(folder, "0.jpg")
    image_path1 = os.path.join(folder, "1.jpg")
    image_path2 = os.path.join(folder, "2.jpg")

    # Λήψη χαρακτηριστικών για τις τρεις εικόνες
    features0 = get_deep_features(image_path0)
    features1 = get_deep_features(image_path1)
    features2 = get_deep_features(image_path2)

    # Υπολογισμός ομοιοτήτων
    similarity_01 = 1 - distance.cosine(features0, features1)
    similarity_02 = 1 - distance.cosine(features0, features2)
    folder_name = os.path.basename(folder)

    # Τύπωση των αποτελεσμάτων
    print(f"Folder: {folder_name}, SIMILARITY (0,1): {similarity_01:.4f}, SIMILARITY (0,2): {similarity_02:.4f}")
