# Boltzman Machines

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

# Converting ratings to binary
# Training Set
training_set_matrix[training_set_matrix == 0] = -1
training_set_matrix[training_set_matrix == 1] = 0
training_set_matrix[training_set_matrix == 2] = 0
training_set_matrix[training_set_matrix >= 3] = 1
# Test Set
test_set_matrix[test_set_matrix == 0] = -1
test_set_matrix[test_set_matrix == 1] = 0
test_set_matrix[test_set_matrix == 2] = 0
test_set_matrix[test_set_matrix >= 3] = 1
