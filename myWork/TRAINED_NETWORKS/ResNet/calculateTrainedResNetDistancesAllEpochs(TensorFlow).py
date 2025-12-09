import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model, Model
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
    # Χρησιμοποιούμε το reshape για να μετατρέψουμε τα χαρακτηριστικά σε 2D πίνακες (flatten)
    features1 = features1.flatten().reshape(1, -1)
    features2 = features2.flatten().reshape(1, -1)
    
    similarity = cosine_similarity(features1, features2)
    distance = 1 - similarity[0][0]
    return distance

def calculate_distances_for_all_epochs(base_dir: str, output_file: str, model_dir: str, num_epochs: int = 25):
    """Υπολογίζει τις αποστάσεις για όλα τα cases και τα μοντέλα από όλες τις εποχές και τις αποθηκεύει σε ένα αρχείο Excel."""
    results = {'Case': []}  # Προσθέτουμε τη στήλη Case

    # Υπολογισμός των αποστάσεων μόνο μία φορά για τα επίπεδα χωρίς βάρη
    print("Processing constant layers...")
    constant_results = {
        'conv5_block3_out_cosine': [],
        'conv5_block3_out_avg_cosine': [],
        'global_average_pooling2d': []
    }

    for case in sorted(os.listdir(base_dir)):
        case_dir = os.path.join(base_dir, case)
        if os.path.isdir(case_dir):
            print(f"Processing constant layers for case: {case}")  # Εκτύπωση του τρέχοντος case
            results['Case'].append(case)  # Προσθέτουμε το case στη στήλη
            images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]

            # Χρησιμοποιούμε ένα μοντέλο για τα σταθερά επίπεδα
            model_path = os.path.join(model_dir, f'resnet_model_epoch_01.keras')  # Τυχαίο μοντέλο για σταθερά επίπεδα
            base_model = load_model(model_path)

            # conv5_block3_out και global_average_pooling2d
            conv5_out_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)
            avgpool_model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)

            # Conv5_block3_out αποστάσεις (Cosine Similarity)
            features1_conv5_out = extract_features(images[0], conv5_out_model)
            features2_conv5_out = extract_features(images[1], conv5_out_model)
            features3_conv5_out = extract_features(images[2], conv5_out_model)

            distance_conv5_out_1 = calculate_distance(features1_conv5_out.flatten().reshape(1, -1), features2_conv5_out.flatten().reshape(1, -1))
            distance_conv5_out_2 = calculate_distance(features1_conv5_out.flatten().reshape(1, -1), features3_conv5_out.flatten().reshape(1, -1))
            avg_distance_conv5_out = (distance_conv5_out_1 + distance_conv5_out_2) / 2

            constant_results['conv5_block3_out_cosine'].append(avg_distance_conv5_out)

            # Conv5_block3_out αποστάσεις (Avg Cosine Similarity)
            avg_cos_dist_conv5_1 = 1 - avg_cosine_similarity(features1_conv5_out, features2_conv5_out)
            avg_cos_dist_conv5_2 = 1 - avg_cosine_similarity(features1_conv5_out, features3_conv5_out)
            avg_sp_cos_dist_conv5 = (avg_cos_dist_conv5_1 + avg_cos_dist_conv5_2) / 2

            constant_results['conv5_block3_out_avg_cosine'].append(avg_sp_cos_dist_conv5)

            # Global Average Pooling αποστάσεις
            features1_avgpool = extract_features(images[0], avgpool_model)
            features2_avgpool = extract_features(images[1], avgpool_model)
            features3_avgpool = extract_features(images[2], avgpool_model)

            distance_avgpool_1 = calculate_distance(features1_avgpool, features2_avgpool)
            distance_avgpool_2 = calculate_distance(features1_avgpool, features3_avgpool)
            avg_distance_avgpool = (distance_avgpool_1 + distance_avgpool_2) / 2

            constant_results['global_average_pooling2d'].append(avg_distance_avgpool)

    # Υπολογισμός αποστάσεων για τα βάρη που αλλάζουν ανά epoch
    for epoch in range(1, num_epochs + 1):
        print(f"Processing variable layers for epoch {epoch}...")

        # Φόρτωση του μοντέλου για την τρέχουσα εποχή
        if epoch < 10:
            model_path = os.path.join(model_dir, f'resnet_model_epoch_0{epoch}.keras')
        else:
            model_path = os.path.join(model_dir, f'resnet_model_epoch_{epoch}.keras')

        print(f"Loading model from {model_path}")
        base_model = load_model(model_path)

        # conv5_block3_3_conv, conv5_block3_3_bn, dense_1 (που περιέχουν βάρη)
        conv5_3_conv_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_3_conv').output)
        conv5_3_bn_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_3_bn').output)
        fc_model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_1').output)

        for case in sorted(os.listdir(base_dir)):
            case_dir = os.path.join(base_dir, case)
            if os.path.isdir(case_dir):
                images = [os.path.join(case_dir, f"{i}.jpg") for i in range(3)]

                # Conv5_block3_3_conv αποστάσεις
                features1_conv5_conv = extract_features(images[0], conv5_3_conv_model)
                features2_conv5_conv = extract_features(images[1], conv5_3_conv_model)
                features3_conv5_conv = extract_features(images[2], conv5_3_conv_model)

                distance_conv5_conv_1 = calculate_distance(features1_conv5_conv.flatten().reshape(1, -1), features2_conv5_conv.flatten().reshape(1, -1))
                distance_conv5_conv_2 = calculate_distance(features1_conv5_conv.flatten().reshape(1, -1), features3_conv5_conv.flatten().reshape(1, -1))
                avg_distance_conv5_conv = (distance_conv5_conv_1 + distance_conv5_conv_2) / 2

                results[f'epoch_{epoch:02d}_conv5_block3_3_conv'] = results.get(f'epoch_{epoch:02d}_conv5_block3_3_conv', []) + [avg_distance_conv5_conv]

                # Conv5_block3_3_bn αποστάσεις
                features1_conv5_bn = extract_features(images[0], conv5_3_bn_model)
                features2_conv5_bn = extract_features(images[1], conv5_3_bn_model)
                features3_conv5_bn = extract_features(images[2], conv5_3_bn_model)

                distance_conv5_bn_1 = calculate_distance(features1_conv5_bn, features2_conv5_bn)
                distance_conv5_bn_2 = calculate_distance(features1_conv5_bn, features3_conv5_bn)
                avg_distance_conv5_bn = (distance_conv5_bn_1 + distance_conv5_bn_2) / 2

                results[f'epoch_{epoch:02d}_conv5_block3_3_bn'] = results.get(f'epoch_{epoch:02d}_conv5_block3_3_bn', []) + [avg_distance_conv5_bn]

                # Fully Connected αποστάσεις
                features1_fc = extract_features(images[0], fc_model)
                features2_fc = extract_features(images[1], fc_model)
                features3_fc = extract_features(images[2], fc_model)

                distance_fc_1 = calculate_distance(features1_fc, features2_fc)
                distance_fc_2 = calculate_distance(features1_fc, features3_fc)
                avg_distance_fc = (distance_fc_1 + distance_fc_2) / 2
                
                # Αποθήκευση των αποστάσεων για το πλήρως συνδεδεμένο επίπεδο (Fully Connected Layer)
                results[f'epoch_{epoch:02d}_Fully_Connected'] = results.get(f'epoch_{epoch:02d}_Fully_Connected', []) + [avg_distance_fc]

                print(f"Epoch {epoch}, Case {case} - Conv5_block3_3_conv Distance: {avg_distance_conv5_conv:.4f}, Conv5_block3_3_bn Distance: {avg_distance_conv5_bn:.4f}, Fully Connected Distance: {avg_distance_fc:.4f}")

    # Συνδυάζουμε τα αποτελέσματα με τα σταθερά επίπεδα
    results['conv5_block3_out_cosine'] = constant_results['conv5_block3_out_cosine']
    results['conv5_block3_out_avg_cosine'] = constant_results['conv5_block3_out_avg_cosine']
    results['global_average_pooling2d'] = constant_results['global_average_pooling2d']

    # Συνδυάζουμε τα αποτελέσματα με τα σταθερά επίπεδα
    for key in constant_results:
        results[key] = constant_results[key]

    # Μετατροπή των αποτελεσμάτων σε DataFrame και αποθήκευση σε αρχείο Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


# Παράδειγμα χρήσης
if __name__ == "__main__":
    base_dir = r"C:\Users\steli\DIPLOMA\bcc"
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\trained_ResNet_Distances_All_Epochs(TensorFlow).xlsx"
    model_dir = r"D:\Diploma\ResNet_TensorFlow_models_ISIC"
    calculate_distances_for_all_epochs(base_dir, output_file, model_dir)

    