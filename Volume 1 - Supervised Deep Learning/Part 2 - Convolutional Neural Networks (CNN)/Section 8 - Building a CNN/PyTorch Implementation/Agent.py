from CNN import CNN
import torch as T

class Agent():
    def __init__(self, train_data, test_data, lr=1e-3, epochs=1):
        self.train_data = train_data
        self.test_data = test_data
        self.lr = lr
        self.epochs = epochs
        self.cnn = CNN(lr)

    def learn(self):
        all_loss = []
        all_accuracy = []
        for ep in range(1, self.epochs+1):
            print(f'Epoch {ep}')
            epoch_loss = 0
            epoch_accuracy = 0
            for x_batch, y_batch in self.train_data:
                x_batch, y_batch = x_batch.to(self.cnn.device), y_batch.to(self.cnn.device)
                self.cnn.optimizer.zero_grad()
                y_pred = self.cnn.forward(x_batch.type(T.float))
                loss = self.cnn.loss(y_pred, y_batch.unsqueeze(1).type(T.float))
                accuracy = self.accuracy(y_pred, y_batch.unsqueeze(1).type(T.float))
                loss.backward()
                self.cnn.optimizer.step()
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
            all_loss.append(epoch_loss/len(self.train_data))
            all_accuracy.append(epoch_accuracy/len(self.train_data))
            test_accuracy, test_loss = self.evaluate()
            print(f'Train Loss:{epoch_loss/len(self.train_data)} | Train Accuracy:{epoch_accuracy/len(self.train_data)} | Test Loss:{test_loss} | Test Accuracy:{test_accuracy}')
        return all_loss, all_accuracy

    def accuracy(self, y_pred, y_correct):
        y_pred = T.round(y_pred)
        correct_sum = (y_pred == y_correct).sum().float()
        return correct_sum/y_correct.shape[0]
    
    def evaluate(self):
        tot_accuracy = 0
        tot_loss = 0
        for x_test, y_test in self.test_data:
            x, y = x_test.to(self.cnn.device), y_test.to(self.cnn.device)
            y_pred = self.cnn.forward(x)
            tot_accuracy += self.accuracy(y_pred, y.unsqueeze(1))
            tot_loss += self.cnn.loss(y_pred, y.unsqueeze(1).type(T.float))
        return tot_accuracy/len(self.test_data), tot_loss/len(self.test_data)
