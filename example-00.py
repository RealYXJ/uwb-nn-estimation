import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the length of sequences to be generated
sequence_length = 10
latent_dim = 100

# Generator
generator = keras.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Reshape((1, latent_dim)),
    layers.LSTM(128, return_sequences=True),
    layers.TimeDistributed(layers.Dense(1))
])

# Discriminator
discriminator = keras.Sequential([
    layers.Input(shape=(sequence_length, 1)),
    layers.LSTM(128, return_sequences=True),
    layers.TimeDistributed(layers.Dense(1)),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# GAN
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = keras.models.Model(gan_input, gan_output)

# Compile models
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
batch_size = 32
epochs = 1000

for epoch in range(epochs):
    # Generate random noise as input to the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # Generate synthetic sequences
    generated_sequences = generator.predict(noise)
    
    # Real sequences (for demonstration, you would use your real data here)
    real_sequences = np.random.rand(batch_size, sequence_length, 1)
    
    # Concatenate real and generated sequences for discriminator training
    combined_sequences = np.concatenate([real_sequences, generated_sequences])
    
    # Labels for discriminator (1 for real, 0 for generated)
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    
    # Train discriminator
    d_loss = discriminator.train_on_batch(combined_sequences, labels)
    
    # Generate new random noise for the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # Labels for generator (1 for trying to fool discriminator)
    labels = np.ones((batch_size, 1))
    
    # Train GAN
    g_loss = gan.train_on_batch(noise, labels)
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

# After training, you can generate sequences using the generator
generated_sequences = generator.predict(np.random.normal(0, 1, (10, latent_dim)))
print("Generated Sequences:")
print(generated_sequences)
