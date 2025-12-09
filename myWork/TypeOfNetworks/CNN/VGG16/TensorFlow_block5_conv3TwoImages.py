import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Καταστολή προειδοποιήσεων TensorFlow

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)  # Καταστολή προειδοποιήσεων Keras και TensorFlow

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Φόρτωση του προεκπαιδευμένου VGG16 μοντέλου
base_model = VGG16(weights='imagenet', include_top=False)

# Επιλογή του επιπέδου από το οποίο θα πάρεις τα χαρακτηριστικά
layer_name = 'block5_conv3'
intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_deep_features(img_path):
    img_array = preprocess_image(img_path)
    features = intermediate_layer_model.predict(img_array)
    features = features.flatten()  # Επίπεδοποίηση του πίνακα για ευκολότερη σύγκριση
    return features

# Παράδειγμα χρήσης:
image_path1 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
image_path2 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"

features1 = get_deep_features(image_path1)
features2 = get_deep_features(image_path2)

# Τύπωσε το μέγεθος και τον τύπο των feature maps
print("Shape of feature_maps1:", features1.shape)
print("Type of feature_maps1:", type(features1))

# Υπολογισμός της cosine similarity
cosine_sim = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

# Υπολογισμός της απόστασης ως 1 - cosine similarity
distance = 1 - cosine_sim
print(f'Η απόσταση μεταξύ των δύο εικόνων είναι: {distance}')
