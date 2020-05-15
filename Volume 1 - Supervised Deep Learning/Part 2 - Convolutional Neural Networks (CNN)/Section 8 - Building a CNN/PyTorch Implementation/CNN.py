import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):

    def __init__(self, lr=1e-3):
        super().__init__()
        self.convl1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pooling1 = nn.MaxPool2d(2)
        self.convl2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pooling2 = nn.MaxPool2d(2)
        self.convl3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.pooling3 = nn.MaxPool2d(4)
        inputs = self._get_connected_layer_input()
        self.fcl1 = nn.Linear(inputs, 256)
        #self.drop_out1 = nn.Dropout(p=0.4)
        self.fcl2 = nn.Linear(256, 128)
        #self.drop_out2 = nn.Dropout(p=0.4)
        self.output = nn.Linear(128, 1)

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.BCELoss()

        self.device = T.device(T.cuda.current_device()
                               if T.cuda.is_available() else "cpu")
        self.to(self.device)

        print(f'Neural Net Initialized. Using device : {self.device} !')

    def _get_connected_layer_input(self):
        inp = T.zeros(1, 3, 64, 64)
        op = self.convl1(inp)
        op = self.pooling1(op)
        op = self.convl2(op)
        op = self.pooling2(op)
        op = self.convl3(op)
        op = self.pooling3(op)
        return op.view(-1, 1).shape[0]

    def forward(self, data):
        op = F.relu(self.convl1(data))
        op = self.pooling1(op)
        op = F.relu(self.convl2(op))
        op = self.pooling2(op)
        op = F.relu(self.convl3(op))
        op = self.pooling3(op)
        op = op.view(op.size()[0], -1)
        op = F.relu(self.fcl1(op))
        #op = self.drop_out1(op)
        op = F.relu(self.fcl2(op))
        #op = self.drop_out2(op)
        op = T.sigmoid((self.output(op)))
        return op
