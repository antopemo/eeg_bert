from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

from BERT.models import Bert_25_channels, Bert_64_channels
from aux_func.data_preprocess import Preprocessor
import tensorflow as tf
from tensorflow import keras

import datetime

# print(tf.version)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

batch_size = 16
window_width = 256
window_steps = 1
channels = []
#channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]

# Define the checkpoints folder
checkpoint_path = "./checkpoints/train"

bert_config_file = os.path.join("BERT", "bert_config.json")

learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="SparseCatAc")

out_shape = [window_width, len(channels), 1] if channels else [window_width, 64, 1]

if __name__ == "__main__":
    dataset, train_dataset, test_dataset, val_dataset = Preprocessor(batch_size,
                                                                     window_width,
                                                                     window_steps,
                                                                     prueba=0,
                                                                     limpio=0,
                                                                     paciente=1,
                                                                     channels=channels,
                                                                     transpose=True,
                                                                     output_shape=out_shape
                                                                     ).classification_tensorflow_dataset()
    if len(channels) == 25:
        model = Bert_25_channels.create_model(tuple(out_shape), adapter_size=None)
    if len(channels) == 0:
        model = Bert_64_channels.create_model(tuple(out_shape), adapter_size=None)

    model.compile(optimizer=optimizer,
                  loss=train_loss,
                  metrics=[train_accuracy, 'acc'])

    model_path = os.path.normpath(
        "./checkpoints/BERT-HigherDropout-64-16-cabezas" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(model_path)
    os.mkdir(model_path + '/training_weights')

    checkpoint_path = os.path.join(model_path, "training_weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    with open(model_path + '/model_architecture.json', 'w') as f:
        json.dump(model.to_json(), f)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_freq='epoch',
                                                     verbose=1)

    log_dir = ".log/eegs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1,
                                                       write_graph=True,
                                                       write_images=True,
                                                       update_freq='epoch',
                                                       profile_batch=2
                                                       )
    total_epoch_count = 10
    model.fit(x=train_dataset,
              validation_data=val_dataset,
              shuffle=True,
              epochs=total_epoch_count,
              callbacks=[keras.callbacks.EarlyStopping(monitor="SparseCatAc", patience=10, restore_best_weights=True),
                         tensorboard_callback, cp_callback])

    model.evaluate(x=test_dataset,
                   callbacks=[tensorboard_callback])
