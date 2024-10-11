import os
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras import models, layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

epochs = 500
batch_size = 32
patience = 10
learning_rate = 0.0002
latent_dim = 100 
img_shape = (256, 256, 3) 
model_path = 'checkpoints/model_gan.keras'
exists = os.path.exists(model_path)

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(img_shape), activation='tanh'))
    model.add(layers.Reshape(img_shape))

    return model

def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def load_data():
    train_dataset = image_dataset_from_directory(
        "dataset/archive/Birds_25/train",
        image_size=(256, 256),
        batch_size=batch_size,
        label_mode=None 
    )
    train_dataset = train_dataset.map(lambda x: (x / 127.5) - 1.0) 
    return train_dataset

generator = build_generator()
discriminator = build_discriminator()

if exists:
    generator = models.load_model(model_path + '_generator')
    discriminator = models.load_model(model_path + '_discriminator')
    print("Modelos carregados.")
else:
    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    z = layers.Input(shape=(latent_dim,))
    img = generator(z)
    discriminator.trainable = False 
    validity = discriminator(img)

    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train_gan(epochs, batch_size, sample_interval=200):
        dataset = load_data()

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for real_imgs in dataset:
            
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                gen_imgs = generator.predict(noise)

                d_loss_real = discriminator.train_on_batch(real_imgs[:batch_size], valid)
                d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = combined.train_on_batch(noise, valid)

                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

                if epoch % sample_interval == 0:
                    sample_images(epoch)

        generator.save(model_path + '_generator')
        discriminator.save(model_path + '_discriminator')

    def sample_images(epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5 

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"images/epoch_{epoch}.png")
        plt.close()


    train_gan(epochs=epochs, batch_size=batch_size)


    callback_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, monitor='loss', mode='min', save_best_only=True)
    ]
