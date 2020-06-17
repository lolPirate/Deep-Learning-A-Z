# Case Study (Hybrid Model)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show


# Unsupervised Model

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
frauds = np.concatenate((mappings[(4, 5)], mappings[(6, 5)], mappings[(5, 8)]), axis=0)
frauds = scaling.inverse_transform(frauds)

# Supervised learning

# Creating features
is_fraud = np.array([1 if i[0] in frauds else 0 for i in x])
customers = dataset.iloc[:, 1:].values

# Feature scaling
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Building the ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=len(customers[0])))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(customers, is_fraud, batch_size=10, epochs=3)

# Predicting the probabilitis of fraud

y_pred = model.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]