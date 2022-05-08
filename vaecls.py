import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from imgcls import img
import os
from loadimage import loadimg

imgobj=img()

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

#encoder
encoder_inputs = keras.Input(shape=(256,256, 1)) #28, 28, 1
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#print()


encoder.summary()

#decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(64 * 64 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((64, 64, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            print(z_mean)
            print(z_log_var)
            print(z)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def plot_latent_space(vae, n=256, figsize=15): #n=30, figsize=15
    # display a n*n 2D manifold of digits
    digit_size = 256 #28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            #figure[
             #   i * digit_size : (i + 1) * digit_size,
              #  j * digit_size : (j + 1) * digit_size,
            #] = digit
            figure=digit

            plt.figure(figsize=(figsize, figsize))
            start_range = digit_size // 2
            end_range = n * digit_size + start_range
            plt.imshow(figure, cmap="Greys_r")
            #plt.show()
            plt.savefig(os.path.join(r'/home/ababelrmah/vae_pp/data/vir_generated','figure_'+str(i)+str(j)+'.jpeg'))
            plt.close()

if __name__=="__main__":

    image_size=(256,256)
    train_size = 6000 #60000
    batch_size = 15 #32
    test_size = 1000 #10000
    path=r'/home/ababelrmah/vae_pp/data/viral pneumonia' #r'/home/ababelrmah/nn_ppml/PriMIA/data/train'
    path2=r'/home/ababelrmah/nn_ppml/PriMIA/data/test'
  
    #train_images=imgobj.load_data(os.path.join(path),batch_size,image_size)
    #test_images=imgobj.load_data(os.path.join(path2,'bacterial pneumonia'),batch_size,image_size)
    train_images=loadimg(os.path.join(path),image_size)
    #print(train_images)
    #train_images=imgobj.preprocess(train_images)
    #test_images=imgobj.preprocess(test_images)
  
    trn = np.concatenate([train_images], axis=0)
    trn = np.expand_dims(trn, -1).astype("float32") / 255
    #normalized_ds = x_train.map(lambda x, y: normalization_layer(x))
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(trn,y=None, epochs=30, batch_size=128) #mnist_digits
    plot_latent_space(vae)
