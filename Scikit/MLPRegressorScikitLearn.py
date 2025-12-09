from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Φόρτωση dataset Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Δημιουργία και εκπαίδευση του MLPRegressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', random_state=42)
mlp_regressor.fit(X_train, y_train)

# Προβλέψεις στα δεδομένα ελέγχου
y_pred = mlp_regressor.predict(X_test)

# Υπολογισμός Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error MLPRegressor: {mse}')