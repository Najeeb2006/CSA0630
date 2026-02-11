import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X, y)
lin_pred = lin_model.predict(X)

# Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# Output
print("Actual Values      :", y)
print("Linear Prediction  :", lin_pred.astype(int))
print("Polynomial Prediction:", poly_pred.astype(int))

# Plot
plt.scatter(X, y)
plt.plot(X, lin_pred)
plt.plot(X, poly_pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.show()
