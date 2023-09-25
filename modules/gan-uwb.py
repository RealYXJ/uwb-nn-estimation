import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generator for Range Error Estimation
class Generator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.generator = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            layers.Dense(output_dim)
        ])

    def call(self, x):
        return self.generator(x)

# Discriminator for Range Error Estimation
class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.discriminator = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        return self.discriminator(x)