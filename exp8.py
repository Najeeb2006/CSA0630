import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area': [800, 1000, 1200, 1500, 1800, 2000],
    'Price': [2000000, 2500000, 3000000, 3800000, 4500000, 5000000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Area']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[1400]])

print("Predicted House Price for 1400 sqft:", prediction[0])
