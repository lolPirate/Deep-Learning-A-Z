import torch as T
import numpy as np
from ANN import ANN

class Agent():
    def __init__(self, input_dims, classes, lr=1e-4):
        self.input_dims = input_dims
        self.classes = classes
        self.lr = lr
        self.ann = ANN(self.input_dims, self.classes, lr=lr)

    
    def learn(self, X, y, batch_size=32, epochs=10):
        self.ann.optimizer.zero_grad()
        X = T.tensor(X).to(self.ann.device)
        y = T.tensor(y, dtype=T.float).to(self.ann.device)

        for ep in range(epochs):
            print("Epoch:", ep)
            for data, label in zip(X, y):
                pred = self.ann.forward(data)
                #pred = 1 if pred>=0.5 else 0
                #pred = T.tensor(pred, dtype=T.float).to(self.ann.device)
                #print(pred, label)
                cost = self.ann.loss(pred, label)
                cost.backward()
                self.ann.optimizer.step()
    
    def evaluate(self, X):
        X = T.tensor(X).to(self.ann.device)
        pred = []
        for data in X:
            pred.append(self.ann.forward(data))
        pred = T.tensor(pred).to(self.ann.device)

        pred = pred>=0.5

        return np.array(pred)
