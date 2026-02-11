import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area': [600, 800, 1000, 1200, 1500, 1800, 2000],
    'Price': [1500000, 2000000, 2600000, 3200000, 4000000, 4600000, 5200000]
}

df = pd.DataFrame(data)

# Feature and target
X = df[['Area']]
y = df['Price']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for 1400 sqft
prediction = model.predict([[1400]])

print("Predicted House Price for 1400 sqft:", int(prediction[0]))
