import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance
import time

# Καταστολή προειδοποιήσεων
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')
tf.get_logger().setLevel('ERROR')

# Ορισμός της διαδρομής του dataset
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
file_path_cpu = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances_Comparison.xlsx"

# Συνάρτηση προεπεξεργασίας εικόνας
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Συνάρτηση εξαγωγής χαρακτηριστικών
def get_deep_features(model, img_path):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)
    features = features.flatten()
    return features

# Συνάρτηση υπολογισμού αποστάσεων για τα cases
def calculate_distances(base_model, layer_name1, layer_name2, device_name):
    print(f"Running on {device_name}")
    
    # Δημιουργία μοντέλων για τα επιλεγμένα επίπεδα
    model_layer1 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name1).output)
    model_layer2 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name2).output)

    case_distances_layer1 = []
    case_distances_layer2 = []
    cases = []

    start_time = time.time()

    for case in sorted(os.listdir(base_dir)):
        case_dir = os.path.join(base_dir, case)
        if os.path.isdir(case_dir):
            images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
            
            # Χαρακτηριστικά για το πρώτο layer
            abnormal_layer1 = get_deep_features
            abnormal_layer1 = get_deep_features(model_layer1, images[0])
            normal1_layer1 = get_deep_features(model_layer1, images[1])
            normal2_layer1 = get_deep_features(model_layer1, images[2])
            dist1_layer1 = distance.cosine(abnormal_layer1, normal1_layer1)
            dist2_layer1 = distance.cosine(abnormal_layer1, normal2_layer1)
            avg_dist_layer1 = (dist1_layer1 + dist2_layer1) / 2
            case_distances_layer1.append(avg_dist_layer1)
            
            # Χαρακτηριστικά για το δεύτερο layer
            abnormal_layer2 = get_deep_features(model_layer2, images[0])
            normal1_layer2 = get_deep_features(model_layer2, images[1])
            normal2_layer2 = get_deep_features(model_layer2, images[2])
            dist1_layer2 = distance.cosine(abnormal_layer2, normal1_layer2)
            dist2_layer2 = distance.cosine(abnormal_layer2, normal2_layer2)
            avg_dist_layer2 = (dist1_layer2 + dist2_layer2) / 2
            case_distances_layer2.append(avg_dist_layer2)
            
            cases.append(case)
            print(f'Case {case}: Layer {layer_name1} Average distance = {avg_dist_layer1:.4f}, Layer {layer_name2} Average distance = {avg_dist_layer2:.4f}')

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time on {device_name}: {duration:.2f} seconds")

    return cases, case_distances_layer1, case_distances_layer2, duration

# Εκτέλεση με χρήση CPU
if True:  # This block will always execute
    # Απενεργοποίηση της GPU
    tf.config.set_visible_devices([], 'GPU')
    
    # Φόρτωση του προεκπαιδευμένου VGG16 μοντέλου
    base_model = VGG16(weights='imagenet', include_top=False)
    
    layer_name1 = 'block5_conv3'
    layer_name2 = 'block5_pool'
    
    cases, case_distances_layer1, case_distances_layer2, cpu_duration = calculate_distances(base_model, layer_name1, layer_name2, 'CPU')

