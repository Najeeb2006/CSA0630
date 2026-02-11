import numpy as np
from sklearn.mixture import GaussianMixture

# Sample data
X = np.array([[1], [2], [3], [8], [9], [10]])

# Create GMM model (2 clusters)
model = GaussianMixture(n_components=2, random_state=0)

# Fit model (EM Algorithm)
model.fit(X)

# Predict cluster labels
labels = model.predict(X)

# Output
print("Data Points :", X.flatten())
print("Cluster Labels :", labels)
print("Cluster Means :", model.means_.flatten())
print("Cluster Variance :", model.covariances_.flatten())
