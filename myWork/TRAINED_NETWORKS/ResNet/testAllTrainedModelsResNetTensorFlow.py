import os
import tensorflow as tf
from tensorflow.keras import models
import pandas as pd

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

# Αξιολόγηση όλων των 25 μοντέλων
model_dir = r"D:\Diploma\ResNet_TensorFlow_models_ISIC"
results = []

for epoch in range(1, 26):
    # Χρησιμοποίησε κατάλληλη μορφή για τα αρχεία
    if epoch < 10:
        model_path = os.path.join(model_dir, f'resnet_model_epoch_0{epoch}.keras')
    else:
        model_path = os.path.join(model_dir, f'resnet_model_epoch_{epoch}.keras')
    
    # Φόρτωση του μοντέλου
    print(f"\nLoading model for epoch {epoch}: {model_path}")
    model = models.load_model(model_path)
    
    # Αξιολόγηση στο test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    # Εκτύπωση των αποτελεσμάτων αμέσως μετά την αξιολόγηση
    print(f"Epoch {epoch} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    results.append((epoch, test_accuracy))

# Αποθήκευση των αποτελεσμάτων σε αρχείο
df = pd.DataFrame(results, columns=["Epoch", "Test Accuracy"])
output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\ResNet_Test_Accuracies_All_Epochs(TensorFlow).xlsx"
df.to_excel(output_file, index=False)
print(f"\nResults saved to {output_file}")
