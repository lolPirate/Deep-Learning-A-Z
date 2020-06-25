# Auto Encoders

# Imports
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn.parallel
import torch.utils.data


# Getting datasets

# movies = pd.read_csv(r'ml-1m\movies.dat', sep="::", header=None,
#                      names=['MovieID', 'Title', 'Genre'], engine='python', encoding='latin-1')
# users = pd.read_csv(r'ml-1m\users.dat', sep="::", header=None,
#                     names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding='latin-1')
# ratings = pd.read_csv(r'ml-1m\ratings.dat', sep="::", header=None,
#                       names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding='latin-1')

# Preparing training and test set

# Training Set (u1.base)
training_set = pd.read_csv(r'ml-100k\u1.base', sep='\t', header=None,
                           names=['user id', 'movie id', 'rating', 'timestamp'])
training_set = np.array(training_set, dtype='int')

# Test Set (u1.test)
test_set = pd.read_csv(r'ml-100k\u1.test', sep='\t', header=None,
                       names=['user id', 'movie id', 'rating', 'timestamp'])
test_set = np.array(test_set, dtype='int')

# Getting total number of users and movies
nb_users = int(np.max((np.max(training_set[:, 0]), np.max(test_set[:, 0]))))
nb_movies = int(np.max((np.max(training_set[:, 1]), np.max(test_set[:, 1]))))

# Converting data to matrix representation


def convert(data):
    new_data = []
    for user in range(1, nb_users + 1):
        movie_id = data[:, 1][data[:, 0] == user]
        movie_ratings = data[:, 2][data[:, 0] == user]
        ratings = np.zeros(nb_movies)
        ratings[movie_id - 1] = movie_ratings
        new_data.append(list(ratings))
    return new_data


training_set_matrix = convert(training_set)
test_set_matrix = convert(test_set)

# Converting to Torch Tensors
device = T.device(T.cuda.current_device() if T.cuda.is_available() else "cpu")
print(f'Using device : {T.cuda.get_device_name(device)}')
training_set_matrix = T.FloatTensor(training_set_matrix).to(device)
test_set_matrix = T.FloatTensor(test_set_matrix).to(device)

# Creating the Auto Encoder architecture

class SAE(nn.Module):
    def __init__(self, nb_movies, lr=1e-2):
        super(SAE, self).__init__()
        self.lr = lr
        self.in_features = nb_movies
        self.out_features = nb_movies
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=20)
        self.enc1 = nn.Linear(20, 10)
        self.enc2 = nn.Linear(10, 5)
        self.enc3 = nn.Linear(5, 10)
        self.enc4 = nn.Linear(10, 20)
        self.out_layer = nn.Linear(20, out_features=self.out_features)

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else'cpu')
        self.to(self.device)

    def forward(self, data):
        y = T.sigmoid(self.fc1(data))
        y = T.sigmoid(self.enc1(y))
        y = T.sigmoid(self.enc2(y))
        y = T.sigmoid(self.enc3(y))
        y = T.sigmoid(self.enc4(y))
        y = self.out_layer(y)
        return y
        
# Training the SAE

def train():
    sae = SAE(nb_movies)
    epochs = 200
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        s = 0.0
        for id_user in range(nb_users):
            sae.optimizer.zero_grad()
            ip = training_set_matrix[id_user]
            ip = ip.view(1, -1)
            target = ip.clone()
            if T.sum(target.data > 0) > 0:
                output = sae(ip)
                target.requires_grad = False
                output[target == 0] = 0
                loss = sae.loss(output, target)
                mean_corrector = nb_movies/float(T.sum(target.data > 0) + 1e-10)
                loss.backward()
                sae.optimizer.step()
                epoch_loss += np.sqrt(loss.item()*mean_corrector)
                s += 1.0
        print(f'Epoch: {epoch:>3} | Train Loss: {epoch_loss/s:>000.2f}')
    T.save(sae, r'models\sae.h5')

# Testing the SAE

def test():
    sae = T.load(r'models\sae.h5')
    loss = 0.0
    s = 0.0
    for id_user in range(nb_users):
            ip = training_set_matrix[id_user]
            ip = ip.view(1, -1)
            target = test_set_matrix[id_user]
            target = target.view(1, -1)
            if T.sum(target.data > 0) > 0:
                output = sae(ip)
                target.requires_grad = False
                output[target == 0] = 0
                loss = sae.loss(output, target)
                mean_corrector = nb_movies/float(T.sum(target.data > 0) + 1e-10)
                loss += np.sqrt(loss.item()*mean_corrector)
                s += 1.0
    print(f'Test Loss: {loss/s}')

#train()
#test()

def sample_test(user_id):
    sae = T.load(r'models\sae.h5')

    test_set_ratings = test_set_matrix[user_id]
    training_set_ratings = training_set_matrix[user_id]
    output = sae(training_set_ratings)

    test_set_ratings = test_set_ratings.cpu().numpy()
    output = output.detach().cpu().numpy()

    output[test_set_ratings == 0] = None
    output = np.round(output)
    movies = pd.read_csv(r'ml-1m\movies.dat', sep="::", header=None, names=['MovieID', 'Title', 'Genre'], engine='python', encoding='latin-1')
    movie_names = movies.iloc[:nb_movies, 1].values
    genre = movies.iloc[:nb_movies, 2].values

    data = {'Movies':movie_names, 'Genre':genre, 'Original Ratings':list(test_set_ratings), 'Predicted Ratings':list(output)}
    df = pd.DataFrame.from_dict(data)
    df.dropna(inplace=True)
    print(df)

sample_test(3)
    







