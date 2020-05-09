import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from Agent import Agent
from DataProcessing import DataProcessing

file_name = 'data\Churn_Modelling.csv'

data_processing = DataProcessing(file_name)

X_train, X_test, y_train, y_test = data_processing.get_data(test_size=0.2)
#print(y_test)

input_dims = len(X_train[0])
classes = 1
agent = Agent(input_dims, classes)
loss = agent.learn(X_train, y_train, batch_size=32, epochs=500)
preds = agent.evaluate(X_test)

act_preds = []

for pred in preds:
    if pred[0]>=0.5: act_preds.append(1)
    else: act_preds.append(0)

act_preds = np.array(act_preds)

accuracy = (len(y_test) - np.count_nonzero(act_preds - y_test))/len(y_test)
print(accuracy)

cm = confusion_matrix(y_test, act_preds)
print(cm)

plt.plot(loss)
plt.show()
