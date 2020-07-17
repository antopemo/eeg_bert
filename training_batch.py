"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Training batch
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Este script se usa como punto de entrada para entrenar diferentes modelos.
Permite seleccionar un tipo de modelo y un tipo de datos, aunque si no se especifica ninguna opción
prueba todos los modelos secuencialmente.

Tiene dos parámetros:
    -modelo: Número del modelo a entrenar.
        0: '64 canales'
        1: '25 canales'
        2: 'Zonas'
    -datos: Que datos usar para entrenar el modelo:
        0: 'FTI'
        1: 'FTD'
        2: 'Both'
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

from BERT.models import Bert_25_channels, Bert_64_channels, Bert_Zones
from BertTraining_Zones import Zone_Trainer
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

import argparse
import datetime
import os
import sys

modelos = {
    0: '64_channels',
    1: '25_channels',
    2: 'Zones'
}

datos_entrenamiento = {
    0: 'FTI',
    1: 'FTD',
    2: 'Both'
}

pacientes = {
    -1:'pre-post',
    0:'control',
    1:'pre',
    2:'post'
}

total_epoch_count = 5
learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="SparseCatAc")


def train_model(modelo, datos, batch_size=16, window_width=256, window_steps=1, patient=1):
    model_path = os.path.normpath(
        f'./checkpoints/{modelos[modelo]}_{datos_entrenamiento[datos]}_{pacientes[patient]}_'
        f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    out_shape = [window_width, 64]
    prueba = -1 if datos == 2 else datos
    if modelo == 2:
        channels = []
        Zone_Trainer(batch_size, window_width, window_steps, channels, out_shape) \
            .train_individually(total_epoch_count, optimizer, train_loss, train_accuracy)
    else:
        channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52,
                    53] if modelo == 1 else []

        model = Bert_64_channels.create_model(tuple(out_shape), adapter_size=None)
        dataset, train_dataset, test_dataset, val_dataset = Preprocessor(batch_size,
                                                                         window_width,
                                                                         window_steps,
                                                                         prueba=prueba,
                                                                         limpio=0,
                                                                         paciente=patient,
                                                                         channels=channels,
                                                                         transpose=True,
                                                                         output_shape=out_shape
                                                                         ).classification_tensorflow_dataset()

        model.compile(optimizer=optimizer,
                      loss=train_loss,
                      metrics=[train_accuracy, 'acc'])

        os.mkdir(model_path)
        os.mkdir(model_path + '\\training_weights')

        checkpoint_path = os.path.join(model_path, "training_weights\\weights.{epoch:02d}-{val_loss:.2f}.hdf5")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        with open(model_path + '\\model_architecture.json', 'w') as f:
            json.dump(model.to_json(), f)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_freq='epoch',
                                                         verbose=1)

        log_dir = ".log\\eegs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                           histogram_freq=1,
                                                           write_graph=True,
                                                           write_images=True,
                                                           update_freq='epoch',
                                                           profile_batch=2
                                                           )
        model.fit(x=train_dataset,
                  validation_data=val_dataset,
                  shuffle=True,
                  epochs=total_epoch_count,
                  callbacks=[
                      keras.callbacks.EarlyStopping(monitor="SparseCatAc", patience=10, restore_best_weights=True),
                      tensorboard_callback, cp_callback])

        model.evaluate(x=test_dataset,
                       callbacks=[tensorboard_callback])


def main(arguments):
    parser = argparse.ArgumentParser('Este script se usa como punto de entrada para entrenar diferentes modelos.'
                                     'Permite seleccionar un tipo de modelo y un tipo de datos, aunque si no se especifica ninguna opción'
                                     'prueba todos los modelos secuencialmente.')
    parser.add_argument('--modelo', help='tipo de modelo a entrenar', type=int)
    parser.add_argument('--datos', help='FTI, FTD, Both', type=int)
    parser.add_argument('--paciente', help='All, control, pre, post', type=int, default=1, choices=list(range(-1,3)))
    args = parser.parse_args(arguments)
    if args.modelo in list(range(3)) and args.datos in list(range(3)):
        print(f"Modelo {modelos[args.modelo]} Datos {datos_entrenamiento[args.datos]}")
        train_model(args.modelo, args.datos)

    elif args.modelo in list(range(3)) and args.datos not in list(range(3)):
        for datos in range(3):
            print(f"Modelo {modelos[args.modelo]} Datos {datos_entrenamiento[datos]}")
            train_model(args.modelo, datos, patient=args.paciente)
    elif args.modelo not in list(range(3)) and args.datos in list(range(3)):
        for modelo in range(3):
            print(f"Modelo {modelos[modelo]} Datos {datos_entrenamiento[args.datos]}")
            train_model(modelo, args.datos, patient=args.paciente)
    else:
        print("Probando todos los modelos")
        for modelo in range(3):
            for datos in range(3):
                print(f"Modelo {modelos[modelo]} Datos {datos_entrenamiento[datos]}")
                train_model(modelo, datos, patient=args.paciente)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
