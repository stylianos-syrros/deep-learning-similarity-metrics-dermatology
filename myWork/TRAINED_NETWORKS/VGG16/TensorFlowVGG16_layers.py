import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Φόρτωση του προεκπαιδευμένου μοντέλου VGG16 χωρίς το fully connected μέρος
base_model = VGG16(weights='imagenet', include_top=False)

# Εκτύπωση των ονομάτων των επιπέδων
for i, layer in enumerate(base_model.layers):
    print(f"Layer {i + 1}: {layer.name}")

# Εκτύπωση του επιπέδου avgpool
avgpool_layer = base_model.get_layer('block5_pool').output
avgpool_layer = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(avgpool_layer)
print("\nAvgpool:")
print(avgpool_layer)

# Εκτύπωση των επιπέδων classifier
print("\nClassifier:")
for i, layer in enumerate(base_model.layers[-3:]):  # Τα τελευταία 3 επίπεδα είναι τα classifier layers
    print(f"Layer {i + 1}: {layer.name}")