import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch as T


class ANN(nn.Module):
    def __init__(self, input_dims, classes, lr=1e-4):
        super().__init__()

        self.lr = lr

        self.fcl1 = nn.Linear(*input_dims, 6)
        self.fcl2 = nn.Linear(6, 6)
        self.output = nn.Linear(6, classes)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else'cpu')
        self.to(self.device)

    def forward(self, data):
        data = T.tensor(data, dtype=T.float).to(self.device)
        layer1 = F.relu(self.fcl1(data))
        layer2 = F.relu(self.fcl2(layer1))
        output = T.sigmoid(self.output(layer2))

        return output
