import os
import pickle
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim


def load_data():
    if os.path.exists("../dataset"):
        print("load data from pickle")
        with open("../dataset/train_X.pkl", 'rb') as f:
            train_X = pickle.load(f)
        with open("../dataset/train_y.pkl", 'rb') as f:
            train_y = pickle.load(f)
        with open("../dataset/valid_X.pkl", 'rb') as f:
            valid_X = pickle.load(f)
        with open("../dataset/valid_y.pkl", 'rb') as f:
            valid_y = pickle.load(f)
        with open("../dataset/test_X.pkl", 'rb') as f:
            test_X = pickle.load(f)
        with open("../dataset/test_y.pkl", 'rb') as f:
            test_y = pickle.load(f)
    else:
        (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()
        cifar_X = np.r_[cifar_X_1, cifar_X_2]
        cifar_y = np.r_[cifar_y_1, cifar_y_2]

        cifar_X = cifar_X.astype('float32') / 255.0
        cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

        train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=5000,
                                                            random_state=42)
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=5000,
                                                              random_state=42)

        os.mkdir("../dataset")
        with open("../dataset/train_X.pkl", 'wb') as f1:
            pickle.dump(train_X, f1)
        with open("../dataset/train_y.pkl", 'wb') as f1:
            pickle.dump(train_y, f1)
        with open("../dataset/valid_X.pkl", 'wb') as f1:
            pickle.dump(valid_X, f1)
        with open("../dataset/valid_y.pkl", 'wb') as f1:
            pickle.dump(valid_y, f1)
        with open("../dataset/test_X.pkl", 'wb') as f1:
            pickle.dump(test_X, f1)
        with open("../dataset/test_y.pkl", 'wb') as f1:
            pickle.dump(test_y, f1)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


def semi_clustering():
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_data()
    train_X, semi_X, train_y, semi_y = train_test_split(train_X, train_y, test_size=10000,
                                                        random_state=42)
    original = semi_y.copy()
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

    train_X = np.concatenate((train_X, semi_X))
    train_y = np.concatenate((train_y, semi_y))

    accurate = 0
    for i, y in enumerate(semi_y):
        if np.array_equiv(y, original[i]):
            accurate += 1
    print('accurate: ', accurate)

    return train_X, train_y, valid_X, valid_y, test_X, test_y


if __name__ == '__main__':
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_data()
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                      input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_X, train_y, epochs=50,
                        validation_data=(test_X, test_y), batch_size=128)
