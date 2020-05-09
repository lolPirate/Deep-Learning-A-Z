from DataProcessing import DataProcessing
from Agent import Agent
import numpy as np

file_name = 'data\Churn_Modelling.csv'

data_processing = DataProcessing(file_name)

X_train, X_test, y_train, y_test = data_processing.get_data()

input_dims = [len(X_train[0])]
classes = 1
agent = Agent(input_dims, classes)
agent.learn(X_train, y_train)
pred = agent.evaluate(X_test)

accuracy = (len(pred)-np.count_nonzero(y_test - pred))/len(pred)
print(accuracy)
