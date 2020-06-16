# Self Organizing Maps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# Importing the dataset
dataset = pd.read_csv(r'Credit_Card_Applications.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
scaling = MinMaxScaler(feature_range=(0, 1))
x = scaling.fit_transform(x)

# Training the SOM
SOM = MiniSom(x=10, y=10, input_len=15)
SOM.random_weights_init(x)
SOM.train_random(x, num_iteration=100, verbose=True)

# Visualizing the results
bone()
pcolor(SOM.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for indx, vector in enumerate(x):
    w = SOM.winner(vector)
    plot(w[0]+0.5, w[1]+0.5, markers[y[indx]], markeredgecolor=colors[y[indx]],
         markerfacecolor='None', markersize=10, markeredgewidth=2)
show()

# Finding frauds
mappings = SOM.win_map(x)
frauds = mappings[(7, 8)]
frauds = scaling.inverse_transform(frauds)