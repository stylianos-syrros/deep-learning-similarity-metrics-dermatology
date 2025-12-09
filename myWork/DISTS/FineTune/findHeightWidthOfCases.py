import os
from PIL import Image
import numpy as np

# Αρχικοποίηση λιστών για τα ύψη και τα πλάτη
heights = []
widths = []

# Διαδρομή για τα cases
base_path = r"C:\Users\steli\DIPLOMA\bcc"

# Επανάληψη σε όλα τα cases
for case_number in range(1, 177):
    case_folder = os.path.join(base_path, f"CASE{str(case_number).zfill(3)}")
    
    # Επανάληψη στις 3 εικόνες (0, 1, 2)
    for img_number in range(3):
        img_path = os.path.join(case_folder, f"{img_number}.jpg")
        
        # Άνοιγμα της εικόνας
        with Image.open(img_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

# Υπολογισμός του μέσου όρου των υψών και των πλατών
average_width = int(np.mean(widths))
average_height = int(np.mean(heights))

print(f"Μέσο πλάτος: {average_width}")
print(f"Μέσο ύψος: {average_height}")
