import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam

# Załaduj dane MNIST
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

latent_dim = 100
img_shape = (28, 28, 1)

# ---------------------
# Kod generatora
# ---------------------

def build_generator(latent_dim):
    model = Sequential()

    # Pełna warstwa połączona z 256 neuronami
    model.add(Dense(256, input_dim=latent_dim))
    # Funkcja aktywacji Leaky ReLU
    model.add(LeakyReLU(alpha=0.2))
    # Normalizacja wsadowa dla stabilności treningu
    model.add(BatchNormalization(momentum=0.8))

    # Druga pełna warstwa połączona z 512 neuronami
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Trzecia pełna warstwa połączona z 1024 neuronami
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Warstwa wyjściowa z funkcją aktywacji tanh dla generowania obrazów
    model.add(Dense(784, activation='tanh'))
    # Przekształcenie wyjścia do kształtu obrazu (28x28x1)
    model.add(Reshape((28, 28, 1)))

    return model

# ---------------------
# Reszta modelu GAN
# ---------------------

def build_discriminator(img_shape):
    model = Sequential()

    # Spłaszczamy obraz do wektora
    model.add(Flatten(input_shape=img_shape))

    # Pełna warstwa połączona z 512 neuronami
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    # Druga pełna warstwa połączona z 256 neuronami
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    # Warstwa wyjściowa z funkcją aktywacji sigmoid dla klasyfikacji
    model.add(Dense(1, activation='sigmoid'))

    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False  # Wyłączamy trening dla dyskryminatora w modelu GAN

    # Wejście dla generatora
    gan_input = Input(shape=(latent_dim,))
    # Generowanie obrazu przez generator
    x = generator(gan_input)
    # Ocena wygenerowanego obrazu przez dyskryminator
    gan_output = discriminator(x)
    # Tworzymy model GAN łącząc generator i dyskryminator
    gan = Model(gan_input, gan_output)
    # Kompilujemy model GAN z binary_crossentropy jako funkcją straty i Adam jako optymalizatorem
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return gan

# ---------------------
# Pozostały kod
# ---------------------

# Funkcja do zapisywania wygenerowanych obrazów
def save_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_imgs = generator.predict(noise)
    generated_imgs = 0.5 * generated_imgs + 0.5  # Przeskalowanie obrazów z zakresu [-1, 1] do [0, 1]

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize, sharex=True, sharey=True)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

def train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=1, batch_size=128, sample_interval=50):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        if epoch % sample_interval == 0:
            save_generated_images(epoch, generator)

# Inicjalizacja generatora, dyskryminatora i modelu GAN
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator)

# Trening modelu GAN
epochs = 10000
batch_size = 128
sample_interval = 1000

# Wywołanie funkcji trenującej
train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
