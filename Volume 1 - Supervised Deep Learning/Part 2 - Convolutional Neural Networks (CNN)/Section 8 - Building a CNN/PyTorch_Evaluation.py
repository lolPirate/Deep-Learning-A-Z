from PIL import Image
from PyTorch_Implementation import CNN
import torch as T
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class Predict():
    def __init__(self, path_to_predict_folder):
        self.path = path_to_predict_folder
        self.cnn = CNN()
        self.cnn.load_state_dict(T.load(r'models\3conv_78_76'))
        self.cnn.eval()
        self.transforms = transforms.Compose([ transforms.Resize((64, 64)), transforms.ToTensor(),transforms.Normalize([.5, .5, .5], [.2, .2, .2])])

    def load_image(self, image_name):
        full_path = os.path.join(self.path, image_name)
        image = Image.open(full_path)
        tensor = self.transforms(image).float().to(self.cnn.device)
        tensor = tensor.view(1,*tensor.shape)
        return tensor, image
            
    def predict(self, image_name):
        tensor, img = self.load_image(image_name)
        with T.no_grad():
            prediction = self.cnn(tensor)
        fig = plt.figure()
        imgplot = plt.imshow(img)
        title = 'Dog' if T.round(prediction).item() == 1 else 'Cat'
        plt.title(title)
        plt.xlabel(prediction.item())
        plt.show()

if __name__ == '__main__':
    path = r'dataset/single_prediction/'
    cnn = Predict(path)
    cnn.predict('cat_or_dog_13.jpg')