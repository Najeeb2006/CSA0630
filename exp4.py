import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Layer 1 (Hidden)
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (Output)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        # Calculate error at output
        error_out = y - output
        d_output = error_out * self.sigmoid_derivative(output)

        # Calculate error at hidden layer
        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Update weights and biases (Gradient Descent)
        self.weights2 += self.a1.T.dot(d_output) * learning_rate
        self.weights1 += X.T.dot(d_hidden) * learning_rate
        self.bias2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing with XOR Data (Classic ANN benchmark)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and Train
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the results
print("\nFinal Predictions:")
predictions = nn.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i][0]:.4f} (Actual: {y[i][0]})")