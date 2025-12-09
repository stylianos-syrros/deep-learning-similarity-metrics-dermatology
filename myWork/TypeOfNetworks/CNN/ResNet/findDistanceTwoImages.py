import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import logging
from averageCosineSimilarity import calculate_cosine_similarity as avg_cosine_similarity


# Καταστολή προειδοποιήσεων TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Ελέγξτε αν η GPU είναι διαθέσιμη και θέστε τη συσκευή
if tf.config.list_physical_devices('GPU'):
    print("Using GPU with CUDA")
else:
    print("Using CPU")

def load_and_preprocess_image(img_path: str):
    """Φορτώνει και προεπεξεργάζεται μια εικόνα."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img_path: str, model: Model):
    """Εξάγει τα χαρακτηριστικά από μια εικόνα χρησιμοποιώντας το μοντέλο."""
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return features

def calculate_distance(features1, features2):
    """Υπολογίζει την απόσταση μεταξύ δύο συνόλων χαρακτηριστικών χρησιμοποιώντας cosine similarity."""
    similarity = cosine_similarity(features1, features2)
    distance = 1 - similarity[0][0]
    return distance

def main(img_path1: str, img_path2: str):
    """Κύρια συνάρτηση που φορτώνει το μοντέλο και υπολογίζει την απόσταση μεταξύ δύο εικόνων."""
    base_model = ResNet50(weights='imagenet')

    # Conv5_x features
    conv5_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)
    features1_conv5 = extract_features(img_path1, conv5_model)
    features2_conv5 = extract_features(img_path2, conv5_model)
    print("features from conv5 : ", features1_conv5.shape)
    distance_conv5 = calculate_distance(features1_conv5.flatten().reshape(1, -1), features2_conv5.flatten().reshape(1, -1))
    print(f"Conv5_x distance between {img_path1} and {img_path2}: {distance_conv5}")

    # Average Cosine Distance for Conv5_x features
    avg_cos_dist_conv5 = 1 - avg_cosine_similarity(features1_conv5, features2_conv5)
    print(f"Average(Spatial Positions) Cosine Distance (Conv5_x) between {img_path1} and {img_path2}: {avg_cos_dist_conv5}")

    # Average Pooling features
    avgpool_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    features1_avgpool = extract_features(img_path1, avgpool_model)
    features2_avgpool = extract_features(img_path2, avgpool_model)
    print("features from avgpool : ", features1_avgpool.shape)
    distance_avgpool = calculate_distance(features1_avgpool, features2_avgpool)
    print(f"Average Pooling distance between {img_path1} and {img_path2}: {distance_avgpool}")

    # Fully Connected Layer features
    fc_model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)
    features1_fc = extract_features(img_path1, fc_model)
    features2_fc = extract_features(img_path2, fc_model)
    print("features from fc : ", features1_fc.shape)
    distance_fc = calculate_distance(features1_fc, features2_fc)
    print(f"Fully Connected Layer distance between {img_path1} and {img_path2}: {distance_fc}")

# Παράδειγμα χρήσης
if __name__ == "__main__":
    img_path1 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
    img_path2 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"
    main(img_path1, img_path2)
