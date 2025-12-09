import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
import numpy as np
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
from averageCosineSimilarity import calculate_cosine_similarity as avg_cosine_similarity

# Δημιουργία φίλτρου Hanning (2D)
def create_hanning_window(size):
    hann1d = np.hanning(size)
    hann2d = np.outer(hann1d, hann1d)
    return hann2d / np.sum(hann2d)  # Κανονικοποίηση ώστε να έχει άθροισμα 1

# Συνάρτηση L2 Pooling με Hanning window
def l2_pooling_with_hanning(x):
    # Δημιουργία φίλτρου Hanning για τα χωρικά διανύσματα
    hanning_window = create_hanning_window(x.shape[1])  # Το μέγεθος είναι το ίδιο με το spatial size
    hanning_window = tf.convert_to_tensor(hanning_window, dtype=tf.float32)

    # Εφαρμογή του φίλτρου Hanning στην έξοδο του convolutional layer
    hanning_window = hanning_window[tf.newaxis, :, :, tf.newaxis]  # Προσαρμογή διαστάσεων για συμβατότητα με το batch
    x = x * hanning_window  # Εφαρμογή του παραθύρου σε κάθε κανάλι ξεχωριστά

    # Εφαρμογή L2 Pooling: μέσος όρος των τετραγώνων των τιμών και μετά τετραγωνική ρίζα
    return tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2]))

# Φόρτωση του ResNet50 μοντέλου με προεκπαιδευμένα βάρη
base_model = ResNet50(weights='imagenet', include_top=False)

# Δημιουργία νέου μοντέλου με L2 Pooling και Hanning window (καθορισμένο output_shape)
l2_hanning_pooling_model = Model(
    inputs=base_model.input, 
    outputs=Lambda(l2_pooling_with_hanning, output_shape=(2048,))(base_model.output)
)

# Δημιουργία νέου μοντέλου με Average Pooling
avg_pooling_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Δημιουργία μοντέλου που παίρνει την έξοδο του τελευταίου convolutional layer
conv_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

# Δημιουργία νέου μοντέλου με L2 Pooling
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
    # Αφαίρεση της παρτίδας με το squeeze
    embedding1 = np.squeeze(embedding1)
    embedding2 = np.squeeze(embedding2)
    return cosine(embedding1, embedding2)

def calculate_distance_avg_spatial(embedding1, embedding2):
    # Αφαίρεση της παρτίδας με το squeeze
    embedding1 = np.squeeze(embedding1)
    embedding2 = np.squeeze(embedding2)
    return 1-avg_cosine_similarity(embedding1, embedding2)

# Φόρτωση εικόνων
image_path1 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
image_path2 = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"

# Υπολογισμός embeddings με average pooling
avg_embedding1, avg_embedding2 = get_embeddings(avg_pooling_model, image_path1, image_path2)

# Υπολογισμός embeddings με L2 pooling
l2_embedding1, l2_embedding2 = get_embeddings(l2_pooling_model, image_path1, image_path2)

# Υπολογισμός embeddings για το τελευταίο convolutional layer
conv_embedding1, conv_embedding2 = get_embeddings(conv_layer_model, image_path1, image_path2)

# Υπολογισμός embeddings με L2 pooling με Hanning window
l2_hanning_embedding1, l2_hanning_embedding2 = get_embeddings(l2_hanning_pooling_model, image_path1, image_path2)

# Υπολογισμός απόστασης χρησιμοποιώντας average pooling
avg_distance = calculate_distance(avg_embedding1, avg_embedding2)

# Υπολογισμός απόστασης χρησιμοποιώντας L2 pooling
l2_distance = calculate_distance(l2_embedding1, l2_embedding2)

# Υπολογισμός απόστασης χρησιμοποιώντας average spatial pooling
avg_sp_cos_dist_conv5 = 1 - avg_cosine_similarity(conv_embedding1, conv_embedding2)

# Υπολογισμός απόστασης χρησιμοποιώντας L2 pooling με Hanning window
l2_hanning_distance = calculate_distance(l2_hanning_embedding1, l2_hanning_embedding2)

print(f"Average Pooling Distance: {avg_distance}")
print(f"L2 Pooling Distance: {l2_distance}")
print(f"Average Spatial Pooling Distance: {avg_sp_cos_dist_conv5}")
print(f"L2 Hanning Pooling Distance: {l2_hanning_distance}")
