import glob
import json

from sklearn.metrics import confusion_matrix
from tensorflow import keras

from BertTraining import *
from aux_func.data_preprocess import Preprocessor
from matplotlib import pyplot as plt
import seaborn as sn
from BERT import BertModelLayer
import numpy as np
from aux_func.confussion_matrix import plot_confusion_matrix
import tensorflow as tf

print(tf.version)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if __name__ == "__main__":
    model_path = "C:\\Users\\Ceiec01\\OneDrive - UFV\\PFG\\Codigo\\checkpoints\\BERT-20200225-121605"

    with open(model_path + "\\model_architecture.json", 'r') as f:
        model_arch = json.load(f)

    model = keras.models.model_from_json(model_arch, custom_objects={'BertModelLayer': BertModelLayer})

    prepro = Preprocessor(batch_size,
                          window_width,
                          window_steps,
                          prueba=0,
                          limpio=0,
                          paciente=1,
                          channels=channels,
                          transpose=True,
                          output_shape=out_shape
                          )

    _, _, test_dataset, _ = prepro.classification_tensorflow_dataset()
    train, test, val = prepro.classification_generator_dataset()

    _, y_train = zip(*list(train))
    _, y_test = zip(*list(test))
    _, y_val = zip(*list(val))

    y_train = list(y_train)
    y_test = list(y_test)
    y_val = list(y_val)

    for i, weight in enumerate(glob.glob(model_path + "/training_weights/weights.*.hdf5")):
        model.load_weights(weight)
        y_pred = model.predict(test_dataset, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)

        cf_matrix = confusion_matrix(y_test, y_pred)
        print(cf_matrix)
        plot_confusion_matrix(cm=cf_matrix,
                              normalize=False,
                              target_names=["No Parkinson", "Parkinson"],
                              title="Matriz de confusión",
                              save=model_path + f'\\test_confussion_matrix_{i + 1}.png')

        plot_confusion_matrix(cm=cf_matrix,
                              normalize=True,
                              target_names=["No Parkinson", "Parkinson"],
                              title="Matriz de confusión",
                              save=model_path + f'\\test_confussion_matrix_norm_{i + 1}.png')
