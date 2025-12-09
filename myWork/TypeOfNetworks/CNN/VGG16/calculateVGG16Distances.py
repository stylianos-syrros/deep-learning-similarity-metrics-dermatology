import os
import warnings
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance

# Καταστολή προειδοποιήσεων
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')
tf.get_logger().setLevel('ERROR')

# Ελέγχει αν η GPU είναι διαθέσιμη και θέτει την συσκευή
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Έλεγχος για υποστήριξη CUDA και ενεργοποίησή του
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("CUDA is available and has been enabled.")
    except:
        print("Could not enable CUDA.")
else:
    print("CUDA is not available.")

print("Num GPUs Available: ", len(physical_devices))

# Φόρτωση του προεκπαιδευμένου VGG16 μοντέλου
base_model = VGG16(weights='imagenet', include_top=False)

# Επιλογή των επιπέδων από τα οποία θα πάρεις τα χαρακτηριστικά
layer_name1 = 'block5_conv3'
layer_name2 = 'block5_pool'

# Δημιουργία μοντέλων για τα επιλεγμένα επίπεδα
model_layer1 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name1).output)
model_layer2 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name2).output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_deep_features(model, img_path):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)
    print(f"feature shape: {features.shape}")
    features = features.flatten()
    return features

# Διαδρομή προς τον κατάλογο με τα cases
base_dir = r"C:\Users\steli\DIPLOMA\bcc"
case_distances_layer1 = []
case_distances_layer2 = []
cases = []

# Υπολογισμός απόστασης για κάθε case
# Υπολογισμός απόστασης για κάθε case
for case in sorted(os.listdir(base_dir)):
    case_dir = os.path.join(base_dir, case)
    if os.path.isdir(case_dir):
        images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
        
        # Χαρακτηριστικά για το πρώτο layer
        abnormal_layer1 = get_deep_features(model_layer1, images[0])
        normal1_layer1 = get_deep_features(model_layer1, images[1])
        normal2_layer1 = get_deep_features(model_layer1, images[2])
        
        # Τυπώνουμε τις διαστάσεις των χαρακτηριστικών για το πρώτο layer
        print(f'Case {case}: Features from layer {layer_name1} - Abnormal: {abnormal_layer1.shape}, Normal1: {normal1_layer1.shape}, Normal2: {normal2_layer1.shape}')
        
        # Υπολογισμός απόστασης για το πρώτο layer
        dist1_layer1 = distance.cosine(abnormal_layer1, normal1_layer1)
        dist2_layer1 = distance.cosine(abnormal_layer1, normal2_layer1)
        avg_dist_layer1 = (dist1_layer1 + dist2_layer1) / 2
        case_distances_layer1.append(avg_dist_layer1)
        
        # Χαρακτηριστικά για το δεύτερο layer
        abnormal_layer2 = get_deep_features(model_layer2, images[0])
        normal1_layer2 = get_deep_features(model_layer2, images[1])
        normal2_layer2 = get_deep_features(model_layer2, images[2])
        
        # Τυπώνουμε τις διαστάσεις των χαρακτηριστικών για το δεύτερο layer
        print(f'Case {case}: Features from layer {layer_name2} - Abnormal: {abnormal_layer2.shape}, Normal1: {normal1_layer2.shape}, Normal2: {normal2_layer2.shape}')
        
        # Υπολογισμός απόστασης για το δεύτερο layer
        dist1_layer2 = distance.cosine(abnormal_layer2, normal1_layer2)
        dist2_layer2 = distance.cosine(abnormal_layer2, normal2_layer2)
        avg_dist_layer2 = (dist1_layer2 + dist2_layer2) / 2
        case_distances_layer2.append(avg_dist_layer2)
        
        cases.append(case)
        print(f'Case {case}: Layer {layer_name1} Average distance = {avg_dist_layer1:.4f}, Layer {layer_name2} Average distance = {avg_dist_layer2:.4f}')

# Αποθήκευση των αποτελεσμάτων σε αρχείο Excel
#file_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\VGG16_Distances.xlsx"
#df = pd.DataFrame({
#    'Folder': cases,
#    layer_name1: case_distances_layer1,
#    layer_name2: case_distances_layer2
#})

#df.to_excel(file_path, index=False)
#print("Excel file updated successfully.")
