import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Καταστολή προειδοποιήσεων TensorFlow

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)  # Καταστολή προειδοποιήσεων Keras και TensorFlow

# Επανεκκίνηση του VGG16 μοντέλου για να εφαρμόσεις τις αλλαγές
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance

base_model = VGG16(weights='imagenet', include_top=False)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_deep_features(img_path):
    img_array = preprocess_image(img_path)
    features = intermediate_layer_model.predict(img_array)
    features = features.flatten()
    return features

image_path1 = r"C:\Users\steli\DIPLOMA\images\yeezyCARBON.png"
image_path2 = r"C:\Users\steli\DIPLOMA\images\yeezyONIX.png"

layer_names = [layer.name for layer in base_model.layers]
layer_distances = {}

for layer_name in layer_names:
    intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    try:
        features1 = get_deep_features(image_path1)
        features2 = get_deep_features(image_path2)
        similarity = 1 - distance.cosine(features1, features2)
        layer_distances[layer_name] = similarity
    except Exception as e:
        print(f'Error computing features for {layer_name}: {e}')

for layer_name, similarity in layer_distances.items():
    print(f'{layer_name}: {similarity}')
