import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import pandas as pd
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

def calculate_distances_for_cases(base_dir: str, output_file: str):
    """Υπολογίζει τις αποστάσεις για όλα τα cases και τις αποθηκεύει σε ένα αρχείο Excel."""
    base_model = ResNet50(weights='imagenet')
    
    conv5_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)
    avgpool_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    fc_model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

    results = []

    for case in sorted(os.listdir(base_dir)):
        case_dir = os.path.join(base_dir, case)
        if os.path.isdir(case_dir):
            print(f"Processing case: {case}")
            images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]
            
            features1_conv5 = extract_features(images[0], conv5_model)
            features2_conv5 = extract_features(images[1], conv5_model)
            features3_conv5 = extract_features(images[2], conv5_model)

            distance_conv5_1 = calculate_distance(features1_conv5.flatten().reshape(1, -1), features2_conv5.flatten().reshape(1, -1))
            distance_conv5_2 = calculate_distance(features1_conv5.flatten().reshape(1, -1), features3_conv5.flatten().reshape(1, -1))
            avg_distance_conv5 = (distance_conv5_1 + distance_conv5_2) / 2

            avg_cos_dist_conv5_1 = 1 - avg_cosine_similarity(features1_conv5, features2_conv5)
            avg_cos_dist_conv5_2 = 1 - avg_cosine_similarity(features1_conv5, features3_conv5)
            avg_sp_cos_dist_conv5 = (avg_cos_dist_conv5_1 + avg_cos_dist_conv5_2) / 2

            features1_avgpool = extract_features(images[0], avgpool_model)
            features2_avgpool = extract_features(images[1], avgpool_model)
            features3_avgpool = extract_features(images[2], avgpool_model)

            distance_avgpool_1 = calculate_distance(features1_avgpool, features2_avgpool)
            distance_avgpool_2 = calculate_distance(features1_avgpool, features3_avgpool)
            avg_distance_avgpool = (distance_avgpool_1 + distance_avgpool_2) / 2

            features1_fc = extract_features(images[0], fc_model)
            features2_fc = extract_features(images[1], fc_model)
            features3_fc = extract_features(images[2], fc_model)

            distance_fc_1 = calculate_distance(features1_fc, features2_fc)
            distance_fc_2 = calculate_distance(features1_fc, features3_fc)
            avg_distance_fc = (distance_fc_1 + distance_fc_2) / 2

            results.append({
                'Case': case,
                'Conv5_x Distance': avg_distance_conv5,
                'Average(Spatial Positions) Cosine Distance (Conv5_x)': avg_sp_cos_dist_conv5,
                'Average Pooling Distance': avg_distance_avgpool,
                'Fully Connected Layer Distance': avg_distance_fc
            })
            
            print(f"Case {case} - Conv5_x Distance: {avg_distance_conv5:.4f}, Average(Spatial Positions) Cosine Distance (Conv5_x): {avg_sp_cos_dist_conv5:.4f}, Average Pooling Distance: {avg_distance_avgpool:.4f}, Fully Connected Layer Distance: {avg_distance_fc:.4f}")

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Παράδειγμα χρήσης
if __name__ == "__main__":
    base_dir = r"C:\Users\steli\DIPLOMA\bcc"
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ResNet_Distances.xlsx"
    calculate_distances_for_cases(base_dir, output_file)
