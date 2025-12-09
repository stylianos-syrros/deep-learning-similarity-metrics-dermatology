import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# Φορτώνουμε το μοντέλο ResNet50 με προεκπαιδευμένα βάρη στο ImageNet
model = ResNet50(weights='imagenet')

# Δημιουργούμε ένα εικονικό input για να περάσουμε από το δίκτυο
input_shape = (1, 224, 224, 3)  # Batch size 1, εικόνα 224x224 με 3 κανάλια (RGB)
dummy_input = tf.random.normal(input_shape)

# Εμφανίζουμε τα layers του μοντέλου και τις διαστάσεις των χαρακτηριστικών (features) που παράγει κάθε layer
for layer in model.layers:
    intermediate_model = Model(inputs=model.input, outputs=layer.output)
    intermediate_output = intermediate_model(dummy_input)
    print(f"Layer: {layer.name}, Output Shape: {intermediate_output.shape}")
