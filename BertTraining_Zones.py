from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

from BERT.models import Bert_25_channels, Bert_64_channels, Bert_Zones
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
window_steps = 64
channels = []

class Zone_Trainer:
    def __init__(self, batch_size, window_width, window_steps, channels, out_shape):
        self.dataset, self.train_dataset, self.test_dataset, self.val_dataset = Preprocessor(batch_size,
                                                                                             window_width,
                                                                                             window_steps,
                                                                                             prueba=0,
                                                                                             limpio=0,
                                                                                             paciente=1,
                                                                                             channels=channels,
                                                                                             transpose=True,
                                                                                             output_shape=out_shape
                                                                                             ) \
            .classification_tensorflow_dataset()
        self.model, self.zone_model = Bert_Zones.create_model(tuple(out_shape), adapter_size=None)
        self.model_name = "BERT-Zones-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_path = os.path.normpath("./checkpoints/" + self.model_name)
        os.mkdir(self.model_path)
        os.mkdir(self.model_path + '\\training_weights')

        checkpoint_path = os.path.join(self.model_path, "training_weights\\weights.{epoch:02d}-{val_loss:.2f}.hdf5")

        with open(self.model_path + '\\model_architecture.json', 'w') as f:
            json.dump(self.model.to_json(), f)

        # Create a callback that saves the model's weights
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                              save_freq='epoch',
                                                              verbose=1)

        log_dir = ".log\\eegs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                                histogram_freq=1,
                                                                write_graph=True,
                                                                write_images=True,
                                                                update_freq='epoch',
                                                                profile_batch=2)

    def train_together(self, total_epoch_count, optimizer):
        self.model.compile(optimizer=optimizer,
                           loss=[train_loss, None],
                           metrics=[train_accuracy, 'acc'])

        self.model.fit(x=self.train_dataset,
                       validation_data=self.val_dataset,
                       shuffle=True,
                       epochs=total_epoch_count,
                       callbacks=[
                           keras.callbacks.EarlyStopping(monitor="SparseCatAc", patience=10, restore_best_weights=True),
                           self.tensorboard_callback, self.cp_callback])

    def train_individually(self, total_epoch_count, optimizer, train_loss, train_accuracy):
        for i, zone in enumerate(self.zone_model):
            checkpoint_path = os.path.join(self.model_path,
                                           "training_weights\\weights_" + str(i+1) + ".{epoch:02d}-{val_loss:.2f}.hdf5")

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_freq='epoch',
                                                             verbose=1)
            log_dir = ".log\\eegs\\zone_" + str(i+1) + "\\" + self.model_name
            self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                                    histogram_freq=1,
                                                                    write_graph=True,
                                                                    write_images=True,
                                                                    update_freq='epoch',
                                                                    profile_batch=2)

            zone.compile(optimizer=optimizer,
                         loss=train_loss,
                         metrics=[train_accuracy, 'acc'])
            zone.fit(x=self.train_dataset,
                     validation_data=self.val_dataset,
                     shuffle=True,
                     epochs=total_epoch_count,
                     callbacks=[
                         keras.callbacks.EarlyStopping(monitor="SparseCatAc", patience=10, restore_best_weights=True),
                         cp_callback, self.tensorboard_callback]
                     )
        self.model.save_weights(os.path.join(self.model_path, "training_weights\\weights_9.full.hdf5"))


if __name__ == "__main__":

    # channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]

    learning_rate = 2e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="SparseCatAc")

    out_shape = [window_width, len(channels)] if channels else [window_width, 64]
    epoch_count = 10

    Zone_Trainer(batch_size, window_width, window_steps, channels, out_shape)\
        .train_individually(epoch_count, optimizer, train_loss, train_accuracy)
