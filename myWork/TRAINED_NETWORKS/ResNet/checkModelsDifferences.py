import numpy as np
from tensorflow.keras import models

# Φόρτωση δύο μοντέλων από διαφορετικές εποχές
model_epoch_1 = models.load_model(r"D:\Diploma\ResNet_TensorFlow_models_ISIC\resnet_model_epoch_01.keras")
model_epoch_2 = models.load_model(r"D:\Diploma\ResNet_TensorFlow_models_ISIC\resnet_model_epoch_02.keras")

# Σύγκριση των βαρών
models_are_identical = True  # Αρχικοποιούμε τη σημαία ως True

for layer_1, layer_2 in zip(model_epoch_1.layers, model_epoch_2.layers):
    weights_1 = layer_1.get_weights()
    weights_2 = layer_2.get_weights()
    for w1, w2 in zip(weights_1, weights_2):
        if not np.array_equal(w1, w2):
            models_are_identical = False  # Αν βρούμε διαφορά, αλλάζουμε τη σημαία
            print("Τα μοντέλα είναι διαφορετικά.")
            break  # Σταματάμε τη σύγκριση αν βρούμε διαφορά
    if not models_are_identical:
        break  # Σταματάμε τον εξωτερικό βρόχο αν βρούμε διαφορά

if models_are_identical:
    print("Τα μοντέλα είναι τα ίδια.")
