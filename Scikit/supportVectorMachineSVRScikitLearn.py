from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Φόρτωση dataset Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Δημιουργία του μοντέλου SVM για παλινδρόμηση
svm_regressor_linear = SVR(kernel='linear') # Εδώ χρησιμοποιούμε ένα γραμμικό πυρήνα

# Εκπαίδευση του μοντέλου
svm_regressor_linear.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred_svm_reg_linear = svm_regressor_linear.predict(X_test)

# Υπολογισμός του Mean Squared Error
mse_svm_reg_linear =mean_squared_error(y_test, y_pred_svm_reg_linear)
print(f'Mean Squared Error SVM Linear (Παλινδρόμηση): {mse_svm_reg_linear}')

# Δημιουργία του μοντέλου SVM για παλινδρόμηση
svm_regressor_rbf = SVR(kernel='rbf', C=0.1, gamma='scale') 

# Εκπαίδευση του μοντέλου
svm_regressor_rbf.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred_svm_reg_rbf = svm_regressor_rbf.predict(X_test)

# Υπολογισμός του Mean Squared Error
mse_svm_reg_rbf =mean_squared_error(y_test, y_pred_svm_reg_rbf)
print(f'Mean Squared Error SVM RBF (Παλινδρόμηση): {mse_svm_reg_rbf}')

# Δημιουργία subplot με δύο σειρές και ένα κοινό άξονα (1 για τοποθέτηση δίπλα στο άλλο)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_svm_reg_linear)
plt.xlabel('Πραγματικές Τιμές')
plt.ylabel('Προβλεπόμενες Τιμές')
plt.title('Προβλέψεις SVM για Παλινδρόμηση με linear kernel')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svm_reg_rbf)
plt.xlabel('Πραγματικές Τιμές')
plt.ylabel('Προβλεπόμενες Τιμές')
plt.title('Προβλέψεις SVM για Παλινδρόμηση με rbf kernel')

plt.tight_layout()  # Εξασφαλίζει ότι τα plots δεν θα επικαλύπτονται
plt.show()