from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Φόρτωση dataset Iris
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Δημιουργία ενός απλού Decision Tree Classifier
clf = DecisionTreeClassifier()

# Εκπαίδευση του μοντέλου
clf.fit(X_train, y_train)

# Πρόβλεψη στα δεδομένα ελέγχου
y_pred = clf.predict(X_test)

# Αξιολόγηση της ακρίβειας
accuracy = accuracy_score(y_test, y_pred)
#print(f'Ακρίβεια: {accuracy}')
#print(f"iris: {iris}")
#print(f"len(iris): {len(iris)}")
#print(f"x_test: {X_test}")

#for i in range(len(iris.data)):
#    print("Sample", i+1, ":", iris.data[i], "Class:", iris.target[i])

#for key, value in iris.items():
#    print(key, ":", value)

length_of_data = len(iris['data'])
print(f"Το μέγεθος του 'data' είναι: {length_of_data}")

length_of_target = len(iris['target'])
print(f"Το μέγεθος του 'target' είναι: {length_of_target}")

