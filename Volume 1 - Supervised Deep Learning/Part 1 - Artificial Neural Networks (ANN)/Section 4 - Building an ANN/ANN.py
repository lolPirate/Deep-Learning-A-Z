# imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# getting data
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

# encoding

# categorical independent variables
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
ct_x = ColumnTransformer(
    [("Geography", OneHotEncoder(), [1])], remainder='passthrough')
X = ct_x.fit_transform(X)
X = X[:, 1:]

# splitting to training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# building the ANN

# init ANN
classifier = Sequential()

# first input layer
classifier.add(Dense(6, input_shape=(
    len(X_train[0]),), bias_initializer='uniform', activation='relu'))
# adding drop out
classifier.add(Dropout(rate=0.2))

# first hidden layer
classifier.add(Dense(6, bias_initializer='uniform', activation='relu'))
# adding drop out
classifier.add(Dropout(rate=0.2))

# output layer
classifier.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))

# compiling classifier
classifier.compile(
    optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting to training set
classifier.fit(x=X_train, y=y_train, batch_size=32, epochs=100)

# predicting values
y_pred = classifier.predict(x=X_test)

# metric
y_pred = [1 if i > 0.5 else 0 for i in y_pred]
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : ', cm)
accuracy = (cm[0][0]+cm[1][1])/(sum(cm[0])+sum(cm[1]))
print('Accuracy : ', accuracy)

# predicting a single variable
new_data = np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
new_data = sc.transform(new_data)
print(classifier.predict(new_data))

# predicting a single variable (general method)
new_data = np.array(
    [[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]], dtype=object)
new_data[:, 2] = labelencoder_X_Gender.transform(new_data[:, 2])
new_data = ct_x.transform(new_data)
new_data = new_data[:, 1:]
new_data = sc.transform(new_data)
print(classifier.predict(new_data))

# analysis

# evluating the model (k-fold cross validation)
def build_classifier():
    # init ANN
    classifier = Sequential()
    # first input layer
    classifier.add(Dense(6, input_shape=(
        len(X_train[0]),), bias_initializer='uniform', activation='relu'))
    # adding drop out
    classifier.add(Dropout(rate=0.2))
    # first hidden layer
    classifier.add(Dense(6, bias_initializer='uniform', activation='relu'))
    # adding drop out
    classifier.add(Dropout(rate=0.2))
    # output layer
    classifier.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))
    # compiling classifier
    classifier.compile(
        optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(
    estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
print(accuracies.mean())
print(accuracies.std())

# dropout regularization (overfitting fix)
# this is done to remove correlations

# Parameter tuning

def build_classifier(optimizer):
    # init ANN
    classifier = Sequential()
    # first input layer
    classifier.add(Dense(6, input_shape=(
        len(X_train[0]),), bias_initializer='uniform', activation='relu'))
    # adding drop out
    classifier.add(Dropout(rate=0.2))
    # first hidden layer
    classifier.add(Dense(6, bias_initializer='uniform', activation='relu'))
    # adding drop out
    classifier.add(Dropout(rate=0.2))
    # output layer
    classifier.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))
    # compiling classifier
    classifier.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_classifier)
parameters = {'batch_size': [25, 32, 64], 
            'epochs': [200, 500],
            'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X=X_train, y=y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
print(best_parameters, best_accuracy)
