import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    dataframe = pd.read_csv('data.csv')
    X = dataframe.to_numpy()
    X = np.delete(X, -1, axis=1)
    y = dataframe['WTT_without_Training_nocent'].values
    y = np.eye(11)[y.astype('int32').flatten()]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=5000)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=5000)
    return train_X, train_y, test_X, test_y
