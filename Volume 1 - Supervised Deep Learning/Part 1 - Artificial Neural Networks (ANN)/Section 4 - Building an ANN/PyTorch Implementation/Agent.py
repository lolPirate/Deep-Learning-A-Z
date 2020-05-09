import numpy as np
import torch as T
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ANN import ANN


class Agent():
    def __init__(self, input_dims, classes, lr=1e-5):
        self.input_dims = input_dims
        self.classes = classes
        self.lr = lr
        self.ann = ANN(self.input_dims, self.classes, lr=lr)

    
    def learn(self, data, batch_size=10, epochs=10):
        self.ann.train()
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

        all_loss = []
        all_acc = []
        
        for ep in range(1, epochs+1):
            epoch_loss = 0
            epoch_accuracy = 0
            for x_batch, y_batch in train_loader:
                
                x_batch, y_batch = x_batch.to(self.ann.device), y_batch.to(self.ann.device)
                self.ann.optimizer.zero_grad()

                y_pred = self.ann(x_batch)

                loss = self.ann.loss(y_pred, y_batch.unsqueeze(1).type(T.float))
                accuracy = self.accuracy(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                self.ann.optimizer.step()

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item() 
            
            all_loss.append(epoch_loss/len(train_loader))
            all_acc.append(epoch_accuracy/len(train_loader))
            print(f'Epoch:{ep} | Epoch Loss:{epoch_loss/len(train_loader)} | Epoch Accuracy:{epoch_accuracy/len(train_loader)}')
        return all_loss, all_acc

    def accuracy(self, y_pred, y_test):
        
        y_pred = T.round(y_pred)
        correct_sum = (y_pred == y_test).sum().float()

        return correct_sum/y_test.shape[0]

    
    def evaluate(self, X):
        self.ann.eval()
        test_loader = DataLoader(dataset=X, batch_size=1)
        y_pred_list = []
        with T.no_grad():
            for x_batch in test_loader:
                x_batch = x_batch.to(self.ann.device).type(T.float)
                y_pred = self.ann(x_batch)
                y_pred = T.round(y_pred)
                y_pred_list.append(y_pred)

        return [i.squeeze().tolist() for i in y_pred_list]
