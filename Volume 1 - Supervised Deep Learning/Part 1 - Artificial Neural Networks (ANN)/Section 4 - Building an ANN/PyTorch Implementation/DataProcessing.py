import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataProcessing():
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = None
        self.label_encoder_gender = None
        self.column_transform_x = None
        self.scaling = None
        self.X = None
        self.y = None

    def read_file(self):
        self.df = pd.read_csv(self.file_name)
        self.X = self.df.iloc[:, 3:-1].values
        self.y = self.df.iloc[:, -1].values

    def encode_x(self):
        self.label_encoder_gender = LabelEncoder()
        self.X[:,2] = self.label_encoder_gender.fit_transform(self.X[:, 2])
        self.column_transform_x = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder='passthrough')
        self.X = self.column_transform_x.fit_transform(self.X)
        self.X = self.X[:, 1:]

    def scale_x(self):
        self.scaling = StandardScaler()
        self.X = self.scaling.fit_transform(self.X)

    def split(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)

        return X_train, X_test, y_train, y_test

    def get_data(self, test_size=0.2):
        self.read_file()
        self.encode_x()
        self.scale_x()
        return self.split(test_size=test_size)

    def encode_test_data(self, data):
        data[:, 2] = self.label_encoder_gender.transform(data[:,2])
        data = self.column_transform_x.transform(data)
        data = data[:, 1:]

        return data

    def scale_test_data(self, data):
        data = self.scaling.transform(data)

        return data
