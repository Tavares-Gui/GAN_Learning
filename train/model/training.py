import os
from tensorflow.keras import models, layers, activations, \
optimizers, utils, losses, initializers, metrics, callbacks, regularizers

epochs = 500
batch_size = 32
patience = 10
learning_rate = 0.002
model_path = 'checkponts/model.keras'
exists = os.path.exists(model_path)

model = models.load_model(model_path) \
    if exists \
    else models.Sequential([

    ])

if exists:
    model.summary()
else:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        # loss=losses.SparseCategoricalCrossentropy(),
        # metrics=['']
    )

train = utils.image_dataset_from_directory(
    "dataset/archive/Birds_25/train"
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    image_size=(128, 128),
    batch_size=batch_size
)

test = utils.image_dataset_from_directory(
    "dataset/archive/Birds_25/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=True,
    image_size=(128, 128),
    batch_size=batch_size
)

model.fit(
    train,
    epochs=epochs,
    validation_data=test,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True
        )
    ]
)
