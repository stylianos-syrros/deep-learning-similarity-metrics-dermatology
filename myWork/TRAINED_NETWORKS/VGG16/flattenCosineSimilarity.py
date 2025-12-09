import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(feature_maps1, feature_maps2):    
    # Βεβαιώσου ότι τα feature maps έχουν τις ίδιες διαστάσεις
    assert feature_maps1.shape == feature_maps2.shape, "Τα feature maps πρέπει να έχουν τις ίδιες διαστάσεις"
    
    # Flatten τα feature maps
    v1 = feature_maps1.flatten()
    v2 = feature_maps2.flatten()
        
    # Υπολογισμός του cosine similarity για το ζεύγος διανυσμάτων
    cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

    return cos_sim

# Τα feature maps για τις δύο εικόνες
#A = np.array([[0.2, 0.4, 0.5], [0.6, 0.8, 0.1], [0.3, 0.7, 0.9]])
#B = np.array([[0.4, 0.9, 0.3], [0.7, 0.5, 0.2], [0.8, 0.2, 0.4]])
#C = np.array([[0.7, 0.8, 0.4], [0.6, 0.4, 0.3], [0.5, 0.9, 0.7]])
#D = np.array([[0.5, 0.2, 0.6], [0.9, 0.8, 0.1], [0.2, 0.7, 0.3]])
#E = np.array([[0.3, 0.6, 0.4], [0.5, 0.9, 0.2], [0.8, 0.4, 0.7]])
#F = np.array([[0.8, 0.5, 0.1], [0.3, 0.6, 0.4], [0.7, 0.2, 0.9]])
#G = np.array([[0.6, 0.1, 0.3], [0.4, 0.7, 0.5], [0.2, 0.8, 0.6]])
#H = np.array([[0.7, 0.4, 0.9], [0.3, 0.2, 0.8], [0.6, 0.5, 0.1]])

# Δημιουργία των feature maps για τις δύο εικόνες
#feature_maps1 = np.stack([A, B, C, D])
#feature_maps2 = np.stack([E, F, G, H])

# Τύπωσε το μέγεθος και τον τύπο των feature maps
#print("Shape of feature_maps1:", feature_maps1.shape)
#print("Type of feature_maps1:", type(feature_maps1))

# Υπολογισμός της ομοιότητας
#cosine_sim = calculate_cosine_similarity(feature_maps1, feature_maps2)
#print(f'Η μέση τιμή του cosine similarity είναι: {cosine_sim}')
