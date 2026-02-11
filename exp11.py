import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Income': [30000, 50000, 70000, 20000, 90000, 60000, 40000],
    'Loan': [10000, 20000, 15000, 5000, 30000, 18000, 12000],
    'Credit_Score': [0, 1, 1, 0, 1, 1, 0]  # 0 = Bad, 1 = Good
}

df = pd.DataFrame(data)

# Features and target
X = df[['Income', 'Loan']]
y = df['Credit_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
print("Predicted :", y_pred)
print("Actual    :", y_test.values)
print("Accuracy  :", accuracy_score(y_test, y_pred))
