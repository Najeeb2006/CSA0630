import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN doesn't "train" in the traditional sense; it just stores data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # 1. Compute distances between x and all examples in the training set
        # Using Euclidean Distance: sqrt(sum((x1 - x2)^2))
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Return the most common class label (Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# --- Testing the Implementation ---

# Load Dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit model
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Calculate Accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"K-NN Accuracy: {accuracy * 100:.2f}%")

# Show sample predictions
print("\nSample Predictions (Predicted vs Actual):")
for i in range(5):
    print(f"Predicted: {predictions[i]} | Actual: {y_test[i]}")