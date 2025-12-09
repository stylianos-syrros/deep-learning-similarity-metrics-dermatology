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

# Κατάψυξη όλων των επιπέδων εκτός από τα τελευταία 10 επίπεδα
for layer in base_model.layers[:-10]:  # Παγώνουμε όλα τα επίπεδα εκτός από τα τελευταία 10
    layer.trainable = False

# Τα τελευταία 10 επίπεδα παραμένουν εκπαιδεύσιμα
for layer in base_model.layers[-10:]:
    layer.trainable = True


# Συμπύκνωση του μοντέλου
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Προσαρμοσμένος ModelCheckpoint για αποθήκευση κάθε εποχής
checkpoint_epoch = callbacks.ModelCheckpoint(
    filepath=r'D:\Diploma\ResNet_TensorFlow_models_ISIC\resnet_model_epoch_{epoch:02d}.keras',
    save_weights_only=False,
    save_best_only=False,
    verbose=1
)

checkpoint_best = callbacks.ModelCheckpoint(
    filepath=r'D:\Diploma\ResNet_TensorFlow_models_ISIC\best_resnet_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

csv_logger = callbacks.CSVLogger(r'D:\Diploma\ResNet_TensorFlow_models_ISIC\training_log.csv')

# Callback για να τυπώνει και να αποθηκεύει την val_accuracy και τον χρόνο εκτέλεσης σε κάθε εποχή
class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        epoch_time = time.time() - self.start_time
        print(f"Epoch {epoch + 1}: val_accuracy = {val_accuracy:.4f}, time taken = {epoch_time:.2f} seconds")
        with open(r'D:\Diploma\ResNet_TensorFlow_models_ISIC\val_accuracy_log.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}: val_accuracy = {val_accuracy:.4f}, time taken = {epoch_time:.2f} seconds\n")

custom_callback = CustomCallback()

# Εκπαίδευση του μοντέλου για όλες τις εποχές
num_epochs = 25
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=num_epochs,  # Εκπαίδευση για όλες τις εποχές
    callbacks=[checkpoint_epoch, checkpoint_best, csv_logger, custom_callback]
)

# Αποθήκευση του τελικού μοντέλου
model.save(r'D:\Diploma\ResNet_TensorFlow_models_ISIC\final_resnet_model.h5')

# Φόρτωση του καλύτερου μοντέλου και αξιολόγηση στο test set
best_model = models.load_model(r'D:\Diploma\ResNet_TensorFlow_models_ISIC\best_resnet_model.keras')

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Best model test accuracy: {test_accuracy:.4f}")

print("Training complete. All models and logs have been saved.")
