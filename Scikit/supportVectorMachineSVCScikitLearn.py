from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D


#SVC (Support Vector Classification): Αυτός ο όρος αναφέρεται στη χρήση των SVMs για προβλήματα ταξινόμησης, δηλαδή, όταν θέλουμε να κατηγοριοποιήσουμε τα δεδομένα σε 
#διάφορες κλάσεις. Ο αλγόριθμος επιχειρεί να βρει το υπερεπίπεδο που διαχωρίζει τις διάφορες κλάσεις στον χώρο των χαρακτηριστικών.

# Φόρτωση dataset Iris
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Δημιουργία του μοντέλου SVM
svm_classifier_linear =SVC(kernel='linear') # Εδώ χρησιμοποιούμε ένα γραμμικό πυρήνα

# Εκπαίδευση του μοντέλου
svm_classifier_linear.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred_svm_linear = svm_classifier_linear.predict(X_test)

# Υπολογισμός της ακρίβειας
accuracy_linear = accuracy_score(y_test, y_pred_svm_linear)
print(f'Ακρίβεια SVM με linear: {accuracy_linear}')

# Δημιουργία του μοντέλου SVM
svm_classifier_rbf =SVC(kernel='rbf', C=0.1, gamma='scale') # Εδώ χρησιμοποιούμε ένα γραμμικό πυρήνα

# Εκπαίδευση του μοντέλου
svm_classifier_rbf.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred_svm_rbf = svm_classifier_rbf.predict(X_test)

# Υπολογισμός της ακρίβειας
accuracy_rbf = accuracy_score(y_test, y_pred_svm_rbf)
print(f'Ακρίβεια SVM με rbf: {accuracy_rbf}')

num_dimensions = X_train.shape[1]
print(f"Οι αρχικές διαστάσεις των δεδομένων είναι: {num_dimensions}") # Κάθε δείγμα δεδομένων (ή παρατήρηση) στο σύνολο εκπαίδευσης έχει 4 χαρακτηριστικά

print(set(y_train)) # Διαφορετικές κατηγορίες (κλάσεις) στο σύνολο εκπαίδευσης 

# Προσαρμογή PCA στα δεδομένα για να τα μειώσουμε σε δύο διαστάσεις για να μπορούν να εμφανιστούν γραφικά
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Προβολή των δεδομένων εκπαίδευσης με τις προβλέψεις του γραμμικού SVM
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=50)
plt.title('SVM - Δεδομένα Εκπαίδευσης')

# Προβολή των δεδομένων εκπαίδευσης με τις προβλέψεις του SVM με πυρήνα RBF
X_test_pca = pca.transform(X_test)
plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=50)
plt.title('SVM - Δεδομένα Ελέγχου')

# Προσθήκη επεξήγησης (legend) με τα ονόματα των κλάσεων
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Κλάση 0'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Κλάση 1'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Κλάση 2')]

plt.legend(handles=legend_elements, title="Κλάσεις")
plt.show()