from PIL import Image
from CNN import CNN
import torch as T
from torchvision import transforms
import matplotlib.pyplot as plt
import os


class Predict():
    def __init__(self, path_to_predict_folder):
        self.path = path_to_predict_folder
        self.cnn = CNN()
        self.cnn.load_state_dict(T.load(r'./models/3conv'))
        self.cnn.eval()
        self.transforms = transforms.Compose([transforms.Resize(
            (64, 64)), transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.2, .2, .2])])

    def load_image(self, image_name):
        full_path = os.path.join(self.path, image_name)
        image = Image.open(full_path)
        tensor = self.transforms(image).float().to(self.cnn.device)
        tensor = tensor.view(1, *tensor.shape)
        return tensor, image

    def predict(self, image_name):
        tensor, img = self.load_image(image_name)
        with T.no_grad():
            prediction = self.cnn(tensor)
            print(prediction)
        fig = plt.figure()
        imgplot = plt.imshow(img)
        title = 'Dog' if T.round(prediction).item() == 1 else 'Cat'
        plt.title(title)
        #prediction_percent = round(abs(.5-prediction.item())*100/.5)
        #plt.xlabel(f'{prediction_percent}% probability of {title}')
        plt.show()


if __name__ == '__main__':
    path = r'./dataset/single_prediction/'
    cnn = Predict(path)
    for i in range(1,11):
        cnn.predict(f'cat_or_dog_{i}.jpg')
