import glob
import json

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

from BertTraining import batch_size, window_width, window_steps, channels, out_shape
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
                          paciente=2,
                          channels=channels,
                          transpose=True,
                          output_shape=out_shape
                          )

    test = prepro.test_set

    model.load_weights(glob.glob(model_path + "/training_weights/weights.*.hdf5")[-1])
    y_pred = []
    for x_test, y_test in zip(test[0], test[1]):
        test_dataset = prepro.tf_from_generator([x_test], [y_test])
        pred = model.predict(test_dataset, verbose=1)
        y_pred.append(np.mean(pred, axis=0))

    y_pred = np.argmax(np.asarray(y_pred), axis=1)

    cf_matrix = confusion_matrix(test[1], y_pred)

    with open(model_path + '/classification_report.txt', 'w') as f:
        print(classification_report(test[1], y_pred, target_names=["No Parkinson", "Parkinson"]), file=f)
    print(cf_matrix)

    plot_confusion_matrix(cm=cf_matrix,
                          normalize=False,
                          target_names=["No Parkinson", "Parkinson"],
                          title="Matriz de confusión",
                          save=model_path + f'\\test_eeg_postlevo.png')

    plot_confusion_matrix(cm=cf_matrix,
                          normalize=True,
                          target_names=["No Parkinson", "Parkinson"],
                          title="Matriz de confusión",
                          save=model_path + f'\\test_eeg_norm_postlevo.png')
