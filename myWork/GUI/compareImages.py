from PIL import Image
import numpy as np

def compare_images(image_path1, image_path2):
    # Άνοιγμα των εικόνων
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    
    # Μετατροπή των εικόνων σε numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    
    # Έλεγχος αν οι διαστάσεις των εικόνων είναι ίδιες
    if image1_array.shape != image2_array.shape:
        return False
    
    # Σύγκριση των εικόνων
    comparison = np.array_equal(image1_array, image2_array)
    
    return comparison

# Χρήση της συνάρτησης
image_path1 = r"C:\Users\steli\DIPLOMA\myProgramms\GUI\testImages\img0DISTS.png"
image_path2 = r"C:\Users\steli\DIPLOMA\myProgramms\GUI\testImages\img0DISTS.png"
if compare_images(image_path1, image_path2):
    print("Οι εικόνες είναι ίδιες.")
else:
    print("Οι εικόνες δεν είναι ίδιες.")
