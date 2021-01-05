import tensorflow as tf

from keras import layers
from keras.activations import sigmoid

from model.generator import Generator
from model.discriminator import Discriminator
from metrics import summarize

# Model to map a system summary vector to a point in the generator's latent space
class LatentMapper:

    def __init__(self, batch_size, vector_size, discriminator_rating, discriminator_weight=0.5):
        # The number of examples per training step (should be fairly large for this model)
        self.batch_size = batch_size
        # Size of physical data vectors
        self.vector_size = vector_size
        # How the discriminator rates the generator's performance (the generator's loss function)
        self.discriminator_rating = discriminator_rating
        # How important the discriminator's rating should be in mapping points
        self.discriminator_weight = discriminator_weight

    # Creates a new model from scratch
    def new(self):
        # Restore the generator
        self.load_gan()

        mapper = keras.Sequential()

        # Just a bunch of dense layers to map from summary vector to latent vector
        mapper.add(layers.Dense(self.vector_size), activation='sigmoid')
        mapper.add(layers.Dense(6))
        mapper.add(layers.Dense(12, activation='relu'))
        mapper.add(layers.Dense(24, activation='relu'))
        mapper.add(layers.Dense(24, activation='relu'))
        mapper.add(layers.Dense(24, activation='relu'))
        mapper.add(layers.Dense(self.latent_size))
        # The generator used here was trained from z~N(0,1)
        mapper.add(layers.BatchNormalization())

        mapper.optimizer = keras.optimizers.Adam(1e-4)

        self.restored = False
        self.initialize(mapper)

    # Load generator and discriminator
    def load_gan(self):
        # Get the restored generator
        self.G = Generator(0, 0, restore=True)
        # Use generator's latent size
        self.latent_size = self.G.z_length

        # Get the restored discriminator
        self.D = Discriminator(0, restore=True)

    # Restore a saved model
    def restore(self):
        self.load_gan()

        mapper = keras.models.load_model('LatentMapper')

        self.restored = True
        self.initialize(mapper)

    # Initialize a saved or restored network
    def initialize(self, model):
        self.Z = model

    # Save the model's trainable parameters/optimizer
    def save(self):
        keras.models.save_model(self.z, 'LatentMapper')

    # Loss function to evaluate mapping & image quality
    def loss(self, I, s, Goz):
        # Difference between mapped and target points
        mapping_loss = K.mean(I - s)
        # Quality of generator images, as measured by the discriminator
        quality_loss = self.discriminator_weight*K.mean(self.discriminator_rating(Goz, self.D))

        return mapping_loss + quality_loss

    def train(self, epochs, save_interval):
        for epoch in range(epochs):
            start = time.time()

            for i in range(batch_size):
                last = (i == batch_size - 1)
                self.train_step(last)

            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save()

            print('Epoch #{} took {} seconds'.format(epoch + 1, time.time() - start))

    @tf.function
    def train_step(self, last):
        with tf.GradientTape() as tape:
            s = tf.random.uniform([self.batch_size, self.vector_size], minval=-10, maxval=10)
            z = self.Z(s)
            Goz = self.G(z, training=False)
            I = sigmoid(summarize(Goz))

            loss = self.loss(I, s, Goz)

        gradient = tape.gradient(loss, self.Z.trainable_variables)
        self.Z.optimizer.apply_gradients(gradient, self.Z.trainable_variables)
