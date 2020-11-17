from skimage.metrics import structural_similarity as ssim
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from sklearn.cluster import KMeans


def load_data():
    if os.path.exists("dataset"):
        print("load data from pickle")
        with open("dataset/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open("dataset/train_y.pkl", 'rb') as f:
            train_y = pickle.load(f)
        with open("dataset/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open("dataset/valid_y.pkl", 'rb') as f:
            valid_y = pickle.load(f)
        with open("dataset/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open("dataset/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    else:
        (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()
        cifar_X = np.r_[cifar_X_1, cifar_X_2]
        cifar_y = np.r_[cifar_y_1, cifar_y_2]

        cifar_X = cifar_X.astype('float32') / 255.0
        cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

        train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=5000,
                                                            random_state=42)
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=11000,
                                                              random_state=42)

        os.mkdir("dataset")
        with open("dataset/train_X.pkl", 'wb') as f1:
            pickle.dump(train_X, f1)
        with open("dataset/train_y.pkl", 'wb') as f1:
            pickle.dump(train_y, f1)
        with open("dataset/valid_X.pkl", 'wb') as f1:
            pickle.dump(valid_X, f1)
        with open("dataset/valid_y.pkl", 'wb') as f1:
            pickle.dump(valid_y, f1)
        with open("dataset/test_X.pkl", 'wb') as f1:
            pickle.dump(test_X, f1)
        with open("dataset/test_y.pkl", 'wb') as f1:
            pickle.dump(test_y, f1)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


def semi_clustering():
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_data()
    train_X, semi_X, train_y, semi_y = train_test_split(train_X, train_y, test_size=10000,
                                                          random_state=42)
    train_X_1d = np.reshape(train_X, (-1, 3072))
    k = 10
    kmeans = KMeans(n_clusters=k)
    X_digits_dist = kmeans.fit_transform(train_X_1d)
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)
    for i, X in enumerate(semi_X):
        similarity = -1
        d = -1
        for digit in representative_digit_idx:
            s = ssim(X, train_X[digit], multichannel=True)
            if s > similarity:
                d = digit
                similarity = s
        semi_y[i] = train_y[d]

    with open("valid_y_semi.pkl", 'wb') as f1:
        pickle.dump(semi_y, f1)

    train_X = np.concatenate(train_X, semi_X)
    train_y = np.concatenate(train_y, semi_y)

    return train_X, train_y, valid_X, valid_y, test_X, test_y


if __name__ == '__main__':
    with open("dataset/valid_y.pkl", 'rb') as f:
        valid_y = pickle.load(f)
    with open("dataset/valid_y_semi.pkl", 'rb') as f1:
        valid_y_semi = pickle.load(f1)
    error = 5000
    for i, y in enumerate(valid_y_semi):
        if np.array_equiv(y, valid_y[i]):
            error -= 1
    print(error)
    # semi_clustering()
