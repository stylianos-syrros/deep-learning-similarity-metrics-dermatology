import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import canny

# Φόρτωση μιας δείγμα εικόνας
image = data.camera()

# Aλγόριθμο Canny για ανίχνευση ακμών στην εικόνα
# Εφαρμογή φίλτρου Canny για ανίχνευση ακμών
edges = canny(image)

# Δημιουργεί ένα γραφικό παράθυρο με δύο υπο-πλοτς (subplots), ένα για την αρχική εικόνα και ένα για τις ανιχνευμένες ακμές
# Προβολή της αρχικής εικόνας και των ανιχνευμένων ακμών
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

# Μέθοδος ravel() επιστρέφει μια 1D προβολή των υπο-πλοτς, ορίζοντας τον πίνακα ax που περιέχει αυτές τις προβολές
ax= axes.ravel()

#ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].imshow(image, cmap=plt.cm.jet)
ax[0].set_title('Original Image')

#ax[1].imshow(edges, cmap=plt.cm.gray)
ax[1].imshow(edges, cmap=plt.cm.jet)
ax[1].set_title('Edges with Canny filter')

plt.show()