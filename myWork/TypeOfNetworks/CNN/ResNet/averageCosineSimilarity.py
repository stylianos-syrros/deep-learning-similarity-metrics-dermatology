import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(feature_maps1, feature_maps2):    
    # Βεβαιώσου ότι τα feature maps έχουν τις ίδιες διαστάσεις
    assert feature_maps1.shape == feature_maps2.shape, "Τα feature maps πρέπει να έχουν τις ίδιες διαστάσεις"
    
    # Αρχικοποίηση της λίστας για αποθήκευση των cosine similarities
    cosine_similarities = []

    # Διατρέχει όλα τα στοιχεία των feature maps
    for i in range(feature_maps1.shape[1]):
        for j in range(feature_maps1.shape[2]):
            # Δημιουργία των διανυσμάτων από τα feature maps
            v1 = feature_maps1[:, i, j]
            v2 = feature_maps2[:, i, j]

            # Υπολογισμός του cosine similarity για το ζεύγος διανυσμάτων
            cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

            # Προσθήκη του cosine similarity στη λίστα
            cosine_similarities.append(cos_sim)

    # Υπολογισμός του μέσου όρου των cosine similarities
    avg_cos_sim = np.mean(cosine_similarities)
    
    return avg_cos_sim

# Παράδειγμα χρήσης της συνάρτησης
#A = np.array([[0.2, 0.4],
#              [0.6, 0.8]])
#B = np.array([[0.4, 0.9],
#              [0.7, 0.5]])
#C = np.array([[0.7, 0.8],
#              [0.6, 0.4]])
#D = np.array([[0.5, 0.2],
#              [0.9, 0.8]])

# Δημιουργία των feature maps για τις δύο εικόνες
#feature_maps1 = np.stack([A, B])
#feature_maps2 = np.stack([C, D])

# Υπολογισμός της ομοιότητας
#cosine_similarity = calculate_cosine_similarity(feature_maps1, feature_maps2)
#print(f'Η μέση τιμή του cosine similarity είναι: {cosine_similarity}')

# Παράδειγμα χρήσης της συνάρτησης
A = np.array([[0.2, 0.4, 0.5],
              [0.6, 0.8, 0.1],
              [0.3, 0.7, 0.9]])
B = np.array([[0.4, 0.9, 0.3],
              [0.7, 0.5, 0.2],
              [0.8, 0.2, 0.4]])
C = np.array([[0.7, 0.8, 0.4],
              [0.6, 0.4, 0.3],
              [0.5, 0.9, 0.7]])
D = np.array([[0.5, 0.2, 0.6],
              [0.9, 0.8, 0.1],
              [0.2, 0.7, 0.3]])
E = np.array([[0.3, 0.6, 0.4],
              [0.5, 0.9, 0.2],
              [0.8, 0.4, 0.7]])
F = np.array([[0.8, 0.5, 0.1],
              [0.3, 0.6, 0.4],
              [0.7, 0.2, 0.9]])
G = np.array([[0.6, 0.1, 0.3],
              [0.4, 0.7, 0.5],
              [0.2, 0.8, 0.6]])
H = np.array([[0.7, 0.4, 0.9],
              [0.3, 0.2, 0.8],
              [0.6, 0.5, 0.1]])

# Δημιουργία των feature maps για τις δύο εικόνες
#feature_maps1 = np.stack([A, B, C, D])
#feature_maps2 = np.stack([E, F, G, H])

# Τύπωσε το μέγεθος και τον τύπο των feature maps
#print("Shape of feature_maps1:", feature_maps1.shape)
#print("Type of feature_maps1:", type(feature_maps1))

# Υπολογισμός της ομοιότητας
#cosine_similarity = calculate_cosine_similarity(feature_maps1, feature_maps2)
#print(f'Η μέση τιμή του cosine similarity είναι: {cosine_similarity}')