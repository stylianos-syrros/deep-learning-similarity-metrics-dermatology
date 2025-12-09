from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Φόρτωση dataset Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Δημιουργία και εκπαίδευση του MLPClassifier
# Έχουμε δύο κρυφά στρώματα με 10 και 5 νευρώνες αντίστοιχα
# Η παράμετρος max_itter καθορίζει τον μέγιστο αριθμό επαναλήψεων που θα πραγματοποιήσει το μοντέλο κατά την εκπαίδευση. Αν το μοντέλο δεν συγκλίνει (δηλαδή δεν βρίσκει τη βέλτιστη λύση) μετά από αυτόν τον αριθμό επαναλήψεων, η εκπαίδευση θα διακοπε
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500, activation='relu', random_state=42)
mlp_classifier.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred = mlp_classifier.predict(X_test)

# Υπολογισμός ακρίβειας
accuracy = accuracy_score(y_test,y_pred)
print(f'Ακρίβεια MLPClassifier: {accuracy}')

num_categories = len(np.unique(y_train))
print(f"Οι κατηγορίες στο dataset Iris είναι: {num_categories}")

