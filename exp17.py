import numpy as np
from sklearn.cluster import KMeans

# Sample dataset
X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

# Create K-Means model with K = 2
model = KMeans(n_clusters=2, random_state=0)

# Fit model
model.fit(X)

# Predict cluster labels
labels = model.predict(X)

# Output
print("Data Points:\n", X)
print("Cluster Labels:", labels)
print("Centroids:\n", model.cluster_centers_)
