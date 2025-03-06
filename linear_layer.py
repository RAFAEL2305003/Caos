import numpy as np

class LinearLayer:
    def __init__(self, input_size: int, output_size: int, lr):
        self.w = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.lr = lr
    
    def forward(self, x: np.array):
        self.last_x = x
        #self.last_output = self.__sigmoid(np.dot(x, self.w) + self.b)
        print(self.w)
        c = np.dot(x, self.w)	
        print(c)
        #return self.last_output
    
    def backward(self, loss: np.array):
        delta = loss * self.__sigmoid_derivative(self.last_output)
        
        previous_loss = np.dot(delta, self.w.T)
        
        self.w -= self.lr * np.dot(self.last_x.T, delta)
        self.b -= self.lr * np.sum(delta, axis=0, keepdims=True)
        
        return previous_loss
    
    def __sigmoid(self, I):
        return 1 / (1 + np.exp(-I))
    
    def __sigmoid_derivative(self, output):
        return output * (1 - output)

ll = LinearLayer(8, 1, [0, 0, 0, 1, 1, 0, 1, 1])
ll.forward([0, 0, 0, 1, 1, 0, 1, 1])
