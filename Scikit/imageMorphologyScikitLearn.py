from skimage import morphology, io
import matplotlib.pyplot as plt
import numpy as np

# Δημιουργία μιας εικόνας με λευκά και μαύρα κουκκίδες
binary_image = np.zeros((100, 100))
binary_image[30:70, 30:70] = 1

# Εφαρμογή διαδοχικών μορφολογικών λειτουργιών
dilated_image = morphology.binary_dilation(binary_image) # Εφαρμογή μορφολογικής διαστολής (binary dilation) . Η διαστολή επιφέρει το "έντονο" (λευκό) περιεχόμενο να επεκταθεί προς τα έξω
eroded_image = morphology.binary_erosion(binary_image) # Η ερεύνηση είναι η αντίστροφη διαδικασία της διαστολής. Επιφέρει το "έντονο" περιεχόμενο να συρρικνωθεί

# Προβολή των αποτελεσμάτων
plt.imshow(np.concatenate((binary_image, dilated_image, eroded_image), axis=1), cmap='gray')
plt.show()