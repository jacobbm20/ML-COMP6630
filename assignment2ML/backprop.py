import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size)
        self.B1 = np.random.randn(hidden_size, 1)
        self.W2 = np.random.randn(output_size, hidden_size)
        self.B2 = np.random.randn(output_size, 1)
        self.cost_history = []
      
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
          y_hat, Z2, A1 = self.feed_forward(X)
          cost = self.mse_cost(y, y_hat)
          self.cost_history.append(cost)
          dW1, dB1, dW2, dB2 = self.back_propagation(y_hat, Z2, A1)
          self.W1 -= learning_rate * dW1
          self.B1 -= learning_rate * dB1
          self.W2 -= learning_rate * dW2
          self.B2 -= learning_rate * dB2
          if i % 1000 == 0:
            print(f"Epoch: {i}, Cost: {cost}")

        print(f"Final Weights: W1: {self.W1}, B1: {self.B1}, W12: {self.W2}, B2: {self.B2}")

    def feed_forward(self,A0):
        # Input Layer to Hidden Layer Calculations
        Z1 = self.W1 @ A0 + self.B1
        A1 = self.sigmoid(Z1)

        # Hidden Layer to Output Layer Calculations
        Z2 = self.W2 @ A1 + self.B2
        # A2 = relu(Z2)
        A2 = Z2
        y_hat = A2
        return y_hat, Z2, A1

    def mse_cost(self,y, y_hat):
        return np.mean((y_hat - y)**2)

    def sigmoid(self,arr):
        return 1 / (1+np.exp(-1 * arr))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def relu(self,arr):
        return np.maximum(0, arr)

    
    def back_propagation(self, X, y_hat,y, Z2, A1):
        dZ2 = y_hat - y
        dW2 = (1/100) * dZ2 @ A1.T
        dB2 = (1/100) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.sigmoid_derivative(A1)
        dW1 = (1/100) * dZ1 @ X.T
        dB1 = (1/100) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, dB1, dW2, dB2