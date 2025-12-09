import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Δημιουργία τυχαίων δεδομένων
np.random.seed(42)
X = np.random.rand(100, 2)

# Εφαρμογή K-means με K=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Ανάκτηση των κεντροειδών και των κατηγοριών
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Οπτικοποίηση των δεδομένων και των κεντροειδών
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
