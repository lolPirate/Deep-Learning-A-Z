from torch.utils.data import Dataset

class Train_Data(Dataset):

    def __init__(self, X_data, y_data):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class Test_Data(Dataset):

    def __init__(self, X_data):
        super().__init__()
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)