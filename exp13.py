import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Year': [2010, 2012, 2014, 2016, 2018, 2020],
    'Price': [300000, 350000, 400000, 450000, 520000, 600000]
}

df = pd.DataFrame(data)

# Feature and target
X = df[['Year']]
y = df['Price']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for year 2017
prediction = model.predict([[2017]])

print("Predicted Car Price for year 2017:", int(prediction[0]))
