import tensorflow as tf
from tensorflow.keras import models

# Ορισμός του directory του test set
test_dir = r"D:\Diploma\datasets\processed_ISIC_Dataset\test"

# Δημιουργία του data generator για το test set
mean = [0.6459899, 0.52058024, 0.51453681]
std = [0.14522511, 0.15518147, 0.16543107]

def custom_preprocess_input(img):
    img = img / 255.0
    img = (img - mean) / std
    return img

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=custom_preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Φόρτωση του καλύτερου μοντέλου
best_model_path = r'D:\Diploma\ResNet_TensorFlow_models_ISIC\best_resnet_model.keras'
best_model = models.load_model(best_model_path)

# Αξιολόγηση στο test set
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Best model test accuracy: {test_accuracy:.4f}")
