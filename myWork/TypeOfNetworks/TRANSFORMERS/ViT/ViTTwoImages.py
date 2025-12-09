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
    image = Image.open(image_path).convert('RGB')
    #image = image.resize(image_size)
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

def get_deep_features(image_path):
    inputs = preprocess_image(image_path)
    outputs = model(**inputs)
    # Χρησιμοποιούμε την έξοδο από το CLS token
    features = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return features.flatten()

# Παράδειγμα χρήσης:
image_path1 = r"C:\Users\steli\DIPLOMA\bcc\CASE004\1.jpg"
image_path2 = r"C:\Users\steli\DIPLOMA\bcc\CASE004\2.jpg"

features1 = get_deep_features(image_path1)
features2 = get_deep_features(image_path2)

similarity = 1 - distance.cosine(features1, features2)

# Τύπωση της ομοιότητας
print(f'Η ομοιότητα μεταξύ των δύο εικόνων είναι: {similarity}')
