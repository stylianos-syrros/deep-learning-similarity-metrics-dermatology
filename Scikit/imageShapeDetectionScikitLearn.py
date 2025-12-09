from skimage import transform, draw
import matplotlib.pyplot as plt
import numpy as np

# Δημιουργεί μια μαύρη εικόνα (με όλες τις τιμές pixel να είναι μηδενικές) και στη συνέχεια, χρησιμοποιεί την draw.disk για να προσθέσει έναν κύκλο με κέντρο στις συντεταγμένες (35, 35) και ακτίνα 20, θέτοντας τις αντίστοιχες τιμές pixel σε 1.
# Δημιουργία μιας εικόνας με κύκλους
image_with_circles = np.zeros((100, 100))
rr, cc = draw.disk((35, 35), 20)
image_with_circles[rr, cc] = 1

# Εφαρμογή της μετασχηματισμένης μετασχημασίας Hough για ανίχνευση κύκλων
# Εφαρμόζει τον μετασχηματισμό Hough για κύκλους στην εικόνα με την καθορισμένη ακτίνα 20.
hough_circles = transform.hough_circle(image_with_circles, radius=20)
# Χρησιμοποιεί τη μέθοδο hough_circle_peaks για να εντοπίσει τις κορυφές στον μετασχηματισμό Hough και αποθηκεύει τα αποτελέσματα σε έναν πίνακα.
detected_circles = np.array(transform.hough_circle_peaks(hough_circles, radii=np.arange(20, 21)))

# Προβολή της αρχικής εικόνας και των ανιχνευμένων κύκλων
plt.imshow(image_with_circles, cmap='gray')
plt.scatter(detected_circles[:, 1], detected_circles[:, 0], c='red')
plt.show()
