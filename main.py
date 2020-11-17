import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time
import helper

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

bag_size = 128
num_features = 15
noise_dim = 11
num_classes = 11
learning_rate = 1e-3

class generator_model(tf.keras.Model):
    def __init__(self):
        super(generator_model, self).__init__()
        self.dense1 = layers.Dense(16)
        self.dense2 = layers.Dense(16)
        self.dense3 = layers.Dense(num_features)
        self.batch_norm = layers.BatchNormalization()
        self.leaky = layers.LeakyReLU()

    def call(self, X):
        y = self.dense1(X)
        y = self.batch_norm(y)
        y = self.leaky(y)
        y = self.dense2(y)
        y = self.batch_norm(y)
        y = self.leaky(y)
        y = self.dense3(y)
        return y


class discriminator_model(tf.keras.Model):
    def __init__(self):
        super(discriminator_model, self).__init__()
        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(128)
        self.dense3 = layers.Dense(num_classes)
        self.batch_norm = layers.BatchNormalization()
        self.leaky = layers.LeakyReLU()

    def call(self, X):
        x = self.dense1(X)
        x = self.batch_norm(x)
        x = self.leaky(x)
        x = self.dense2(x)
        x = self.batch_norm(x)
        x = self.leaky(x)
        y = self.dense3(x)
        return y, x


def compute_prop(y):
    bag_size = len(y)
    prop = y.sum(axis=0) / bag_size
    return prop


def noise_generator(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator_loss(real_feature, fake_feature):
    return tf.reduce_mean(tf.square(tf.reduce_mean(real_feature, axis=0) - tf.reduce_mean(fake_feature, axis=0)))


def discriminator_loss(prop, real_result, fake_result):
    loss_real = tf.reduce_logsumexp(real_result, axis=1)
    loss_fake = tf.reduce_logsumexp(fake_result, axis=1)
    llp_loss = tf.reduce_mean(
        -tf.reduce_sum(
            tf.cast(prop, tf.float32) * (tf.math.log(tf.reduce_mean(tf.nn.softmax(real_result), [0]) + 1e-7))))
    gan_loss = -tf.reduce_mean(loss_real) + tf.reduce_mean(tf.nn.softplus(loss_real)) + tf.reduce_mean(
        tf.nn.softplus(loss_fake))

    return gan_loss + 1000 * llp_loss
    # return llp_loss


trainX, trainY, testX, testY = helper.load_data()

generator = generator_model()
discriminator = discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)


@tf.function
def train_step(data, prop):
    noise = noise_generator(bag_size, noise_dim)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise)

        real_result, real_feature = discriminator(data)
        fake_result, fake_feature = discriminator(generated_data)

        gen_loss = generator_loss(real_feature, fake_feature)
        disc_loss = discriminator_loss(prop, real_result, fake_result)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


@tf.function
def error(X, Y):
    real_result, real_feature = discriminator(X)
    return 1.0 - tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(real_result, axis=1), tf.argmax(Y, axis=1)), tf.float32))


def compute_error():
    e = 0
    shape = testX.shape[0]
    for n in range(int(shape / bag_size)):
        x = testX[n * bag_size:(n + 1) * bag_size]
        y = testY[n * bag_size:(n + 1) * bag_size]
        e += error(x, y)
    e /= (shape / bag_size)
    return e.numpy()


err_list = []

for epoch in tf.range(500):
    size = trainX.shape[0]
    timer = time.time()
    for i in range(int(size / bag_size)):
        X = trainX[i * bag_size:(i + 1) * bag_size]
        Y = trainY[i * bag_size:(i + 1) * bag_size]
        prop = compute_prop(Y)
        train_step(X, prop)
    err = compute_error()
    err_list.append(err)

    print('')
    print('epoch:', epoch.numpy(), 'time:', time.time() - timer, 'err:', err)
    print(err_list)
