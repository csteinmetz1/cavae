import sys
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import numpy as np
import util
# image generation
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm

n_ch = 3
img_shape = (28, 28, n_ch)
batch_size = 20
latent_dim = 100

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 2, padding='same', activation='relu')(input_img)
x = layers.Conv2D(32, 2, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 2, padding='same', activation='relu')(x)
x = layers.Conv2D(128, 2, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)

	return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])

x = layers.Dense(64, activation='relu')(decoder_input)
x = layers.Dense(6272, activation='relu')(x)
x = layers.Reshape((14, 14, 32))(x)
x = layers.Conv2DTranspose(128, 2, padding='same', activation='relu', strides=(1, 1))(x)
x = layers.Conv2DTranspose(64, 2, padding='same', activation='relu', strides=(1, 1))(x)
x = layers.Conv2DTranspose(32, 2, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(n_ch, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)
decoder.summary()

z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):

	def vae_loss(self, x, z_decoded):
		x = K.flatten(x)
		z_decoded = K.flatten(z_decoded)
		xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
		kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)	
		#kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)

		return xent_loss + kl_loss

	def call(self, inputs):
		x = inputs[0]
		z_decoded = inputs[1]
		loss = self.vae_loss(x, z_decoded)
		self.add_loss(loss, inputs=inputs)
		return x

y = CustomVariationalLayer()([input_img, z_decoded])

vae = Model(input_img, y)
vae.compile(optimizer=Adam(lr=0.001), loss=None)
vae.summary()

print("\nLoading cover art data...")
x_train, x_test = util.load_data(.99)
#(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
#x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
#x_test = x_test.reshape(x_test.shape + (1,))

#x_train = x_train[:10,:,:,:]
#x_test = x_test[:10,:,:,:]

print("\n\nLoaded training data with shape:")
print(x_train.shape)
print(x_test.shape)

def on_epoch_end(epoch, logs):

	n = 15
	figure = np.zeros((28 * n, 28 * n, n_ch))
	for i in range(15):
		for j in range(15):
			z_sample = []
			for dim in range(latent_dim):
				z_sample.append(np.random.normal(0, 1))
			z_sample = np.array(z_sample)
			z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
			x_decoded = decoder.predict(z_sample, batch_size=batch_size)
			cover = x_decoded[0].reshape(28, 28, n_ch)
			figure[i * 28: (i + 1) * 28,
				j * 28: (j + 1) * 28] = cover

	plt.figure(figsize=(10,10))
	plt.imshow(figure, cmap='Greys_r')
	plt.savefig('results/epoch_{}.png'.format(epoch+1))

figure_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# train the thing
vae.fit(x=x_train, y=None,
		shuffle=True,
		epochs=150,
		batch_size=batch_size,
		validation_data=(x_test, None),
		callbacks=[figure_callback])

#n = 15
#cover_size = 28
#figure = np.zeros((cover_size * n, cover_size * n, 3))
#grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
#grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

#for i, yi in enumerate(grid_x):
#	for j, xi in enumerate(grid_y):
#		z_sample = np.array(([xi, yi]))
#		z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
#		x_decoded = decoder.predict(z_sample, batch_size=batch_size)
#		cover = x_decoded[0].reshape(cover_size, cover_size, 3)
#		figure[i * cover_size: (i + 1) * cover_size,
#			   j * cover_size: (j + 1) * cover_size] = cover

#plt.figure(figsize=(10,10))
#plt.imshow(figure)
#plt.show()
#plt.savefig('covers.png')
