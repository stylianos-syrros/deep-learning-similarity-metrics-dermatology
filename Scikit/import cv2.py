import cv2
import matplotlib.pyplot as plt

# Διάβασε μια εικόνα από τον δίσκο
image = cv2.imread("D:\Diploma\ScikitLearn\IMG_6489.jpg")

# Εμφάνισε την εικόνα
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Χωρίσε την εικόνα σε κανάλια BGR
b, g, r = cv2.split(image)

# Υπολόγισε τα ιστογράμματα των καναλιών
hist_b = cv2.calHist([b], [0], None, [256], [0,256])
hist_g = cv2.calHist([g], [0], None, [256], [0,256])
hist_r = cv2.calHist([r], [0], None, [256], [0,256])

# Εμφάνισε τα ιστογράμματα
plt.plot(hist_b, color='blue')
plt.plot(hist_g, color='green')
plt.plot(hist_r, color='red')
plt.title('Color Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

