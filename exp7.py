import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Hours_Studied']]
y = df['Pass']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Predicted Values:", y_pred)
print("Actual Values   :", y_test.values)
print("Accuracy        :", accuracy_score(y_test, y_pred))
