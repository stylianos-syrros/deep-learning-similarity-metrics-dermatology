import os
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
    model_dir = r"D:\Diploma\ResNet_TensorFlow_models_ISIC"

    # Ονόματα αρχείων των μοντέλων
    model_epoch_1_file = os.path.join(model_dir, 'resnet_model_epoch_01.keras')
    model_epoch_2_file = os.path.join(model_dir, 'resnet_model_epoch_02.keras')

    # Εκτύπωση paths των μοντέλων που φορτώνονται
    print(f"Loading model for epoch 1 from: {model_epoch_1_file}")
    print(f"Loading model for epoch 2 from: {model_epoch_2_file}")

    # Φόρτωση μοντέλων
    model_epoch_1 = load_model(model_epoch_1_file)
    model_epoch_2 = load_model(model_epoch_2_file)

    print(f"Comparing models: Epoch 1 vs. Epoch 2")

    # Φορτώνουμε μία εικόνα από το πρώτο case
    images = [load_and_preprocess_image(os.path.join(base_dir, "CASE001", f"{i}.jpg")) for i in range(3)]
    images = np.concatenate(images, axis=0)  # Συνένωση των εικόνων

    # Σύγκριση των εξόδων των layers conv5_block3_3_conv και conv5_block3_3_bn
    compare_layer_outputs(model_epoch_1, model_epoch_2, images, 'conv5_block3_3_conv')
    compare_layer_outputs(model_epoch_1, model_epoch_2, images, 'conv5_block3_3_bn')
