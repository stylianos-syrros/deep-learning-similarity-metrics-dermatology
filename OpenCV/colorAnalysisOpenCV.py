import cv2
import matplotlib.pyplot as plt

# Διάβασε μια εικόνα από τον δίσκο
image = cv2.imread("D:\Diploma\ScikitLearn\IMG_6489.jpg")

# Εμφάνισε την εικόνα
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Χωρίσε την εικόνα σε κανάλια BGR
b, g, r = cv2.split(image)

# Υπολόγισε τα ιστογράμματα των καναλιών
hist_b = cv2.calcHist([b], [0], None, [256], [0,256])
hist_g = cv2.calcHist([g], [0], None, [256], [0,256])
hist_r = cv2.calcHist([r], [0], None, [256], [0,256])

# Εμφάνισε τα ιστογράμματα
plt.plot(hist_b, color='blue')
plt.plot(hist_g, color='green')
plt.plot(hist_r, color='red')
plt.title('Color Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

#Άξονας X (Pixel Value): Αναπαριστά την ένταση του χρώματος. Για μια εικόνα RGB, η ένταση κάθε καναλιού (κόκκινο, πράσινο, μπλε) μπορεί να λάβει τιμές από 0 έως 255.
#Άξονας Y (Frequency): Αναπαριστά τον αριθμό των pixel στην εικόνα που έχουν τη συγκεκριμένη ένταση χρώματος. Το ύψος της καμπύλης σε κάθε σημείο X δείχνει πόσες φορές η συγκεκριμένη ένταση χρώματος εμφανίζεται στην εικόνα.
