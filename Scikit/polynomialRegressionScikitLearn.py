from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Δημιουργία τυχαίων δεδομένων για παλινδρόμηση
np.random.seed(0) #Με τον ορισμό ενός συγκεκριμένου seed (σε αυτή την περίπτωση, το 0), διασφαλίζεις πως κάθε φορά που τρέχεις τον κώδικα, θα παράγονται τα ίδια τυχαία αριθμητικά αποτελέσματα
x = 2 * np.random.rand(100, 1) #Δημιουργεί έναν πίνακα X με 100 γραμμές και 1 στήλη. Κάθε στοιχείο του πίνακα είναι τυχαίος αριθμός που επιλέγεται από μια ομοιόμορφη κατανομή στο διάστημα [0, 1)
y = 4 + 3 * x + np.random.randn(100 , 1)

# Διαχωρισμός των δεδομένων σε σύνολο εκπαίδευσης και ελέγχου
X_train , X_test ,Y_train , Y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)

# Λίστα για να αποθηκεύσει τα MSE για κάθε τιμή του degree
mse_values = []

# Δημιουργία του μοντέλου πολυωνυμικής παλινδρόμησης
for degree in range(2,11):
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Εκπαίδευση του μοντέλου πολυωνυμικής παλινδρόμησης
    polyreg.fit(X_train, Y_train)

    # Προβλέψεις στα δεδομένα ελέγχου
    y_pred_poly = polyreg.predict(X_test)

    # Υπολογισμός του Mean Squared Error
    mse_poly = mean_squared_error(Y_test,y_pred_poly)
    mse_values.append(mse_poly)
    print(f"Mean Squared Error (Polynomial Regression) with degree = {degree}: {mse_poly}")

# Βρες την βέλτιστη τιμή του degree
best_degree = np.argmin(mse_values) + 2

print(f"Η βέλτιστη τιμή του degree είναι: {best_degree}")
print(f"Το Mean Squared Error για την βέλτιστη τιμή του degree είναι: {mse_values[best_degree - 2]}")

# Σχεδίαση των πραγματικών ετικετών και των προβλέψεων με την βέλτιστη τιμή του degree
polyreg = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
polyreg.fit(X_train, Y_train)
y_pred_poly_best = polyreg.predict(X_test)

# Σχεδίαση των πραγματικών ετικετών και των προβλέψεων
plt.scatter(X_test, Y_test, color='black', label='Πραγματικές ετικέτες')
plt.scatter(X_test, y_pred_poly_best, color='red', label='Προβλέψεις (Polynomial Regression)')
plt.title(f'Πολυωνυμική Παλινδρόμηση (Βαθμός {best_degree})')
plt.xlabel('Χαρακτηρηστικά')
plt.ylabel('Ετικέτες')
plt.legend()
plt.show()