from tensorflow import keras

from mapper_model.network import LatentMapper
from model.network import AGAN
from model.discriminator import Discriminator

def rating(Goz, discriminator):
    DoGoz = discriminator(Goz)
    loss = AGAN.loss_G(DoGoz)

    return loss

mapper = LatentMapper(
    batch_size=100,
    vector_size=4,
    discriminator_rating=rating,
    discriminator_weight=1e-3
)
mapper.new()

try:
    mapper.train(
        epochs=500,
        save_interval=50
    )
except Exception as e:
    print(e)
    print('Training paused')
finally:
    print('Saving...')
    aGAN.save()

    print('Clearing session...')
    keras.backend.clear_session()

    quit()
