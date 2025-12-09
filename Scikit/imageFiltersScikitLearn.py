import matplotlib.pyplot as plt
from skimage import filters, img_as_float
from skimage import transform, io, color
import numpy as np

# Διαβάστε την εικόνα από τον δίσκο
image_path = "D:\Diploma\ScikitLearn\IMG_6489.jpg"  # Αντικαταστήστε το path_to_your_image.jpg με το πραγματικό μονοπάτι της εικόνας σας
image = io.imread(image_path) 

# Διαβάστε την εικόνα από τον δίσκο και μετατρέψτε σε grayscale
gray_image = color.rgb2gray(image)

# Εφαρμογή φίλτρου Gaussian για θόλωμα στην αρχική εικόνα
blurred_image = filters.gaussian(gray_image, sigma=5)

# Περιστροφή κατά 90 μοίρες δεξιά στη θολωμένη εικόνα
rotated_blurred_image = transform.rotate(blurred_image, angle=-90, resize=True)

# Περιστροφή κατά 90 μοίρες δεξιά στην αρχική εικόνα
rotated_image = transform.rotate(gray_image, angle=-90, resize=True)

# Κρατήστε μόνο το κοινό μέρος των διαστάσεων
common_height = min(rotated_blurred_image.shape[0], rotated_image.shape[0])
common_width = min(rotated_blurred_image.shape[1], rotated_image.shape[1])

bw_rotated_blurred_image = img_as_float(rotated_blurred_image[:common_height, :common_width])
bw_rotated_image = img_as_float(rotated_image[:common_height, :common_width])

# Εμφάνιση του κοινού μέρους των εικόνων
plt.imshow(np.concatenate((bw_rotated_blurred_image, bw_rotated_image), axis=1), vmin=0, vmax=1, cmap=plt.cm.jet)
plt.show()