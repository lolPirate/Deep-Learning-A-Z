import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ANN(nn.Module):
    def __init__(self, input_dims, classes, lr=1e-5):
        super().__init__()

        self.lr = lr

        self.fcl1 = nn.Linear(input_dims, 6)
        self.drop_out_1 = nn.Dropout(p=0.1)
        self.fcl2 = nn.Linear(6, 6)
        self.drop_out_2 = nn.Dropout(p=0.1)
        self.fcl3 = nn.Linear(6, 6)
        self.output = nn.Linear(6, classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else'cpu')
        self.to(self.device)

    def forward(self, data):
        #data = T.tensor(data, dtype=T.float).to(self.device)
        y = F.relu(self.fcl1(data))
        y = self.drop_out_1(y)
        y = F.relu(self.fcl2(y))
        y = self.drop_out_2(y)
        y = F.relu(self.fcl3(y))
        output = T.sigmoid(self.output(y))

        return output
