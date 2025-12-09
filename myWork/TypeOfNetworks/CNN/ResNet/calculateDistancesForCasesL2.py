import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
import numpy as np
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
from averageCosineSimilarity import calculate_cosine_similarity as avg_cosine_similarity
import os
import pandas as pd

# Δημιουργία φίλτρου Hanning (2D)
def create_hanning_window(size):
    hann1d = np.hanning(size)
    hann2d = np.outer(hann1d, hann1d)
    return hann2d / np.sum(hann2d)  # Κανονικοποίηση ώστε να έχει άθροισμα 1

# Συνάρτηση L2 Pooling με Hanning window
def l2_pooling_with_hanning(x):
    hanning_window = create_hanning_window(x.shape[1])
    hanning_window = tf.convert_to_tensor(hanning_window, dtype=tf.float32)
    hanning_window = hanning_window[tf.newaxis, :, :, tf.newaxis]
    x = x * hanning_window
    return tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2]))

# Φόρτωση του ResNet50 μοντέλου με προεκπαιδευμένα βάρη
base_model = ResNet50(weights='imagenet', include_top=False)

# Δημιουργία των μοντέλων με τις διαφορετικές μεθόδους
avg_pooling_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
l2_hanning_pooling_model = Model(inputs=base_model.input, outputs=Lambda(l2_pooling_with_hanning, output_shape=(2048,))(base_model.output))
conv_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

def l2_pooling(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2]))

l2_pooling_model = Model(inputs=base_model.input, outputs=Lambda(l2_pooling)(base_model.output))

# Φόρτωση και προεπεξεργασία εικόνων
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Υπολογισμός embeddings για δύο εικόνες
def get_embeddings(model, img_path1, img_path2):
    img1 = load_and_preprocess_image(img_path1)
    img2 = load_and_preprocess_image(img_path2)
    
    embedding1 = model.predict(img1)
    embedding2 = model.predict(img2)
    
    return embedding1, embedding2

# Υπολογισμός απόστασης
def calculate_distance(embedding1, embedding2):
    embedding1 = np.squeeze(embedding1)
    embedding2 = np.squeeze(embedding2)
    return cosine(embedding1, embedding2)

def calculate_distance_avg_spatial(embedding1, embedding2):
    embedding1 = np.squeeze(embedding1)
    embedding2 = np.squeeze(embedding2)
    return 1-avg_cosine_similarity(embedding1, embedding2)

# Συνάρτηση που διατρέχει τα cases
def process_all_cases(base_path, num_cases=176):
    results = []
    for case_num in range(1, num_cases + 1):
        case_path = os.path.join(base_path, f"CASE{str(case_num).zfill(3)}")
        img_paths = [os.path.join(case_path, f"{i}.jpg") for i in range(3)]
        
        if all(os.path.exists(p) for p in img_paths):
            # Υπολογισμός αποστάσεων με τους 4 διαφορετικούς τρόπους
            avg_embedding1, avg_embedding2 = get_embeddings(avg_pooling_model, img_paths[0], img_paths[1])
            avg_embedding1, avg_embedding3 = get_embeddings(avg_pooling_model, img_paths[0], img_paths[2])
            avg_distance1 = calculate_distance(avg_embedding1, avg_embedding2)
            avg_distance2 = calculate_distance(avg_embedding1, avg_embedding3)
            avg_distance = (avg_distance1 + avg_distance2) / 2
            
            l2_embedding1, l2_embedding2 = get_embeddings(l2_pooling_model, img_paths[0], img_paths[1])
            l2_embedding1, l2_embedding3 = get_embeddings(l2_pooling_model, img_paths[0], img_paths[2])
            l2_distance1 = calculate_distance(l2_embedding1, l2_embedding2)
            l2_distance2 = calculate_distance(l2_embedding1, l2_embedding3)
            l2_distance = (l2_distance1 + l2_distance2) / 2

            conv_embedding1, conv_embedding2 = get_embeddings(conv_layer_model, img_paths[0], img_paths[1])
            conv_embedding1, conv_embedding3 = get_embeddings(conv_layer_model, img_paths[0], img_paths[2])
            avg_cos_dist_conv5_1 = 1 - avg_cosine_similarity(conv_embedding1, conv_embedding2)
            avg_cos_dist_conv5_2 = 1 - avg_cosine_similarity(conv_embedding1, conv_embedding3)
            avg_sp_cos_dist_conv5 = (avg_cos_dist_conv5_1 + avg_cos_dist_conv5_2) / 2

            l2_hanning_embedding1, l2_hanning_embedding2 = get_embeddings(l2_hanning_pooling_model, img_paths[0], img_paths[1])
            l2_hanning_embedding1, l2_hanning_embedding3 = get_embeddings(l2_hanning_pooling_model, img_paths[0], img_paths[2])
            l2_hanning_distance1 = calculate_distance(l2_hanning_embedding1, l2_hanning_embedding2)
            l2_hanning_distance2 = calculate_distance(l2_hanning_embedding1, l2_hanning_embedding3)
            l2_hanning_distance = (l2_hanning_distance1 + l2_hanning_distance2) / 2

            print(f"Case {case_num}: Average Pooling Distance: {avg_distance}, L2 Pooling Distance: {l2_distance}, Average(Spatial Positions) Cosine Distance (Conv5_x): {avg_sp_cos_dist_conv5}, L2 Hanning Pooling Distance: {l2_hanning_distance}")
            
            results.append({
                "Case": f"CASE{str(case_num).zfill(3)}",
                "Average Pooling Distance": avg_distance,
                "L2 Pooling Distance": l2_distance,
                "Average(Spatial Positions) Cosine Distance (Conv5_x)": avg_sp_cos_dist_conv5,
                "L2 Hanning Pooling Distance": l2_hanning_distance
            })
    
    return results

# Διαδρομή για τα cases
base_path = r"C:\Users\steli\DIPLOMA\bcc"

# Επεξεργασία όλων των cases
results = process_all_cases(base_path)

# Αποθήκευση αποτελεσμάτων σε αρχείο Excel
df = pd.DataFrame(results)
output_path = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ResNetL2poolingDistancesAllCases.xlsx"
df.to_excel(output_path, index=False)

print(f"Αποθήκευση αποτελεσμάτων στο αρχείο {output_path}")
