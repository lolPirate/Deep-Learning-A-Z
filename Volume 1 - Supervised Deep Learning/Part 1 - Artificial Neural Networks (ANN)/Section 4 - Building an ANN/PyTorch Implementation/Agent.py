import numpy as np
import torch as T
from torch.autograd import Variable
import torch.utils.data as data_utils

from ANN import ANN


class Agent():
    def __init__(self, input_dims, classes, lr=1e-5):
        self.input_dims = input_dims
        self.classes = classes
        self.lr = lr
        self.ann = ANN(self.input_dims, self.classes, lr=lr)

    
    def learn(self, X, y, batch_size=10, epochs=10):
        self.ann.optimizer.zero_grad()
        X = T.tensor(X).to(self.ann.device)
        y = T.tensor(y, dtype=T.float).to(self.ann.device)
        train = data_utils.TensorDataset(X, y)
        train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
        average_loss = []
        for ep in range(epochs):
            loss_list = []
            for x_batch, y_batch in train_loader:
                #x_batch = X[beg:beg + batch_size, :]
                #y_batch = y[beg:beg + batch_size]
                y_pred = self.ann.forward(x_batch)
                loss = self.ann.loss(y_pred, y_batch)
                loss.backward()
                self.ann.optimizer.step()
                loss_list.append(loss.item())
            average_loss.append(np.mean(loss_list))
            print(f'Epoch:{ep}\t\tLoss:{np.mean(loss_list)}')
        return average_loss
    
    def evaluate(self, X):
        X = T.tensor(X).to(self.ann.device)
        
        pred = self.ann.forward(X)

        return pred
