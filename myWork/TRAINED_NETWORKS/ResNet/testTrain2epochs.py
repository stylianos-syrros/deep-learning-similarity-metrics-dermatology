import os
import time
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks

# Έλεγχος αν χρησιμοποιείται GPU ή CPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Using GPU: {physical_devices}")
else:
    print("Using CPU")

# Ρυθμίσεις δεδομένων
train_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\train"
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"

# Τιμές mean και std από το training set
mean = [0.6459899, 0.52058024, 0.51453681]
std = [0.14522511, 0.15518147, 0.16543107]

# Κανονικοποίηση με βάση το mean και το std του training set
def custom_preprocess_input(img):
    img = img / 255.0
    img = (img - mean) / std
    return img

# Σταθερό seed για την τυχαιοποίηση
SEED = 123

# Δημιουργία των data generators για train και test sets με σταθερό seed
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,  # Shuffle στο training set
    seed=SEED  # Σταθερό seed για επαναλήψιμη τυχαιοποίηση
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Δεν κάνουμε shuffle στο test set
)

# Δημιουργία του μοντέλου ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Δεν παγώνουμε τα επίπεδα του ResNet50
# Όλα τα επίπεδα είναι εκπαιδεύσιμα
for layer in base_model.layers:
    layer.trainable = True

# Συμπύκνωση του μοντέλου
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Προσαρμοσμένος ModelCheckpoint για αποθήκευση κάθε εποχής
checkpoint_epoch = callbacks.ModelCheckpoint(
    filepath=r'D:\Diploma\ResNet_TensorFlow_models_ISIC(TEST)\resnet_model_epoch_{epoch:02d}.keras',
    save_weights_only=False,
    save_best_only=False,
    verbose=1
)

# Εκπαίδευση του μοντέλου για δύο εποχές
num_epochs = 2
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=num_epochs,  # Εκπαίδευση για 2 εποχές
    callbacks=[checkpoint_epoch]
)

# Αποθήκευση του τελικού μοντέλου
model.save(r'D:\Diploma\ResNet_TensorFlow_models_ISIC(TEST)\final_resnet_model.h5')

# ------------------------------------------
# Σύγκριση των εξόδων των layers σε 2 εποχές
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_and_preprocess_image(img_path: str):
    """Φορτώνει και προεπεξεργάζεται μια εικόνα."""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def compare_layer_outputs(model1, model2, images, layer_name):
    """Συγκρίνει τις εξόδους ενός συγκεκριμένου layer για δύο μοντέλα."""
    layer_model1 = Model(inputs=model1.input, outputs=model1.get_layer(layer_name).output)
    layer_model2 = Model(inputs=model2.input, outputs=model2.get_layer(layer_name).output)

    features1_model1 = layer_model1.predict(images)
    features1_model2 = layer_model2.predict(images)

    # Εκτύπωση των εξόδων
    print(f"Layer {layer_name} output for model 1 (epoch 1): {features1_model1.flatten()[:10]}")  # εκτύπωση των πρώτων 10 τιμών
    print(f"Layer {layer_name} output for model 2 (epoch 2): {features1_model2.flatten()[:10]}")  # εκτύπωση των πρώτων 10 τιμών

    # Έλεγχος αν οι εξόδοι είναι ίδιες
    if np.array_equal(features1_model1, features1_model2):
        print(f"The outputs of {layer_name} are identical for both models.")
    else:
        print(f"The outputs of {layer_name} are different for the models.")

if __name__ == "__main__":
    base_dir = r"C:\Users\steli\DIPLOMA\bcc"
    model_dir = r"D:\Diploma\ResNet_TensorFlow_models_ISIC(TEST)"

    # Φόρτωση μοντέλων
    model_epoch_1 = load_model(os.path.join(model_dir, 'resnet_model_epoch_01.keras'))
    model_epoch_2 = load_model(os.path.join(model_dir, 'resnet_model_epoch_02.keras'))

    # Φορτώνουμε μία εικόνα από το πρώτο case
    images = [load_and_preprocess_image(os.path.join(base_dir, "CASE001", f"{i}.jpg")) for i in range(3)]
    images = np.concatenate(images, axis=0)  # Συνένωση των εικόνων

    # Σύγκριση των εξόδων των layers conv5_block3_3_conv και conv5_block3_3_bn
    compare_layer_outputs(model_epoch_1, model_epoch_2, images, 'conv5_block3_3_conv')
    compare_layer_outputs(model_epoch_1, model_epoch_2, images, 'conv5_block3_3_bn')
