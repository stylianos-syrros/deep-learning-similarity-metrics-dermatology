from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np 

# Δημιουργία τυχαίων δεδομένων για παλινδρόμηση
np.random.seed(0) #Με τον ορισμό ενός συγκεκριμένου seed (σε αυτή την περίπτωση, το 0), διασφαλίζεις πως κάθε φορά που τρέχεις τον κώδικα, θα παράγονται τα ίδια τυχαία αριθμητικά αποτελέσματα
x = 2 * np.random.rand(100, 1) #Δημιουργεί έναν πίνακα X με 100 γραμμές και 1 στήλη. Κάθε στοιχείο του πίνακα είναι τυχαίος αριθμός που επιλέγεται από μια ομοιόμορφη κατανομή στο διάστημα [0, 1)
y = 4 + 3 * x + np.random.randn(100 , 1)

# Διαχωρισμός των δεδομένων σε σύνολο εκπαίδευσης και ελέγχου
X_train , X_test ,Y_train , Y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)

# Δημιουργία του μοντέλου γραμμικής παλινδρόμησης
linear_arg = LinearRegression()

# Εκπαίδευση του μοντέλου
linear_arg.fit(X_train, Y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred = linear_arg.predict(X_test)

# Υπολογισμός του Mean Squared Error
mse = mean_squared_error(Y_test,y_pred)
print(f"Mean Squared Error: {mse}")

# Σχεδίαση των πραγματικών ετικετών και των προβλέψεων
plt.scatter(X_test, Y_test, color ='black', label='Πραγματικές ετικέτες')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Προβλέψεις')
plt.title('Γραμμική Παλινδρόμηση')
plt.xlabel('Χαρακτηριστικά')
plt.ylabel('Ετικέτες')
plt.legend()
plt.show()