import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# Generate synthetic data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)  # Adjust parameters as needed
svm.fit(X)

# Predict outliers/anomalies
y_pred = svm.predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, edgecolors='k')

plt.title('One-Class SVM for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

