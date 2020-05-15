import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch as T

from Agent import Agent
from CustomDatasets import Test_Data, Train_Data
from DataProcessing import DataProcessing

file_name = 'data\Churn_Modelling.csv'

data_processing = DataProcessing(file_name)

X_train, X_test, y_train, y_test = data_processing.get_data(test_size=0.2)


train_data = Train_Data(T.Tensor(X_train), T.tensor(y_train))
test_data = Test_Data(T.tensor(X_test))

params = {'batch_size':25, 'epochs':50}

input_dims = len(X_train[0])
classes = 1

agent = Agent(input_dims, classes, lr=1e-3)

loss, acc = agent.learn(train_data, **params)

y_pred = agent.evaluate(test_data)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))


plt.plot(loss, label="loss", color='r')
plt.plot(acc, label="acc", color='g')
plt.legend()
plt.show()
