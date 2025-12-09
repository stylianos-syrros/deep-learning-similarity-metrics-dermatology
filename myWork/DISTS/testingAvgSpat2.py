import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(self, feature_maps1, feature_maps2):
    assert feature_maps1.shape == feature_maps2.shape, "Τα feature maps πρέπει να έχουν τις ίδιες διαστάσεις"
    
    # Αφαίρεση της διάστασης του batch size (το batch size είναι 1)
    feature_maps1 = feature_maps1[0]  # Νέο σχήμα: (height, width, channels)
    feature_maps2 = feature_maps2[0]
    # Δημιουργία ενός numpy array με σχήμα (batch_size, height, width, channels)
    # Μεατατροπή του σε (height, width, channels)
    cosine_similarities = []

    # Διατρέχει το height και το width
    for i in range(feature_maps1.shape[0]):  # Διατρέχει το height
        for j in range(feature_maps1.shape[1]):  # Διατρέχει το width
            # Δημιουργία των διανυσμάτων από τα feature maps
            v1 = feature_maps1[i, j]  # Διανυσμα για όλα τα κανάλια στη θέση (i, j)
            v2 = feature_maps2[i, j]

            # Υπολογισμός του cosine similarity για το ζεύγος διανυσμάτων
            cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

            # Προσθήκη του cosine similarity στη λίστα
            cosine_similarities.append(cos_sim)

    # Υπολογισμός του μέσου όρου των cosine similarities
    avg_cos_sim = np.mean(cosine_similarities)
    
    return avg_cos_sim
