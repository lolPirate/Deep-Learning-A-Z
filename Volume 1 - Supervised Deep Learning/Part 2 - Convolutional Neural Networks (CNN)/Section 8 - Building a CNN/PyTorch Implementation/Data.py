import torch.utils.data as data
import torchvision
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

class Data():
    def __init__(self, train_path, test_path, batch_size):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.class_indx = None

    def get_train_data(self):
        train_data = torchvision.datasets.ImageFolder(root=self.train_path, transform=transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.2, .2, .2])
        ]))
        self.class_indx = train_data.class_to_idx
        train_data_loader = data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return train_data_loader

    def get_test_data(self):
        test_data = torchvision.datasets.ImageFolder(root=self.test_path, transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.2, .2, .2])
        ]))
        test_data_loader = data.DataLoader(
            test_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return test_data_loader