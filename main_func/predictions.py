from sklearn.metrics import confusion_matrix, classification_report

from main_func.BertTraining import batch_size
from aux_func.data_preprocess import Preprocessor
import numpy as np
from aux_func.confussion_matrix import plot_confusion_matrix
import tensorflow as tf
import os
# print(tf.version)
from aux_func.load_model import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def test_model(model_path, conjunto, patient, prueba=1, combination='mean', mode=0):
    """
    Función que prueba el modelo especificado.
    Esta funcion genera los reportes de clasificacion y matrices de confusion tanto para
    los eegs completos como para los fragmentos. Ademas si el modelo es por zonas tambien genera
    los valores para estas.

    :param model_path: ruta del modelo (La carpeta donde se almacena el modelo)
    :param conjunto: número del 0 al 4: 'test', 'train', 'val', 'full', 'test_pre_post'
    :param patient: número del -1 al 2: All, control, pre, post
    :param prueba: prueba del dataset:
        -1: "Both",
        0: "FTD",
        1: "FTI",
        2: "Resting"
    :param combination: 'mean' o 'majority_voting' para la combinación de los resultados en el
        modelo de zonas
    :param mode: 0: full y chunks, 1 full, 2 chunks
    :return: nada
    """
    # out_shape = [window_width, 64]
    model = load_model(model_path)
    conjuntos = ['test', 'train', 'val', 'full', 'test_pre_post']
    patients = ['control', 'pre', 'post', 'Pre-Post']
    pruebas = {-1: "Both",
               0: "FTD",
               1: "FTI",
               2: "Resting"}
    base_path = f'{conjuntos[conjunto]}_{patients[patient]}_{pruebas[prueba]}'
    zones = False
    if 'Zone' in model_path:
        base_path = f'{base_path}_{combination}'
        zones = True
    print(f'{base_path}')

    channels = []
    if model.input_shape[2] == 25:
        channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]
    out_shape = list(model.input_shape[1:])

    test_post = False
    if conjunto == 4:
        test_post = True

    prepro = Preprocessor(batch_size,
                          model.input_shape[1],
                          64,
                          prueba=prueba,
                          limpio=0,
                          paciente=patient,
                          channels=channels,
                          transpose=True,
                          output_shape=out_shape,
                          test_post=test_post,
                          shuffle=False)

    try:
        os.mkdir(f'{model_path}/{base_path}')
    except Exception:
        pass

    if mode != 2:
        try:
            os.mkdir(f'{model_path}/{base_path}/full_eeg')
        except Exception:
            pass
        if conjunto == 0 or conjunto == 4:
            data = prepro.test_set
        if conjunto == 1:
            data = prepro.train_set
        if conjunto == 2:
            data = prepro.val_set
        if conjunto == 3:
            data = prepro.dataset
        y_pred = []
        y_pred_zones = [[] for _ in range(8)]
        for x_data, y_data in zip(data[0], data[1]):
            test_dataset = prepro.tf_from_generator([x_data], [y_data])
            pred = model.predict(test_dataset, verbose=1)
            if zones:
                pred, zone_pred = pred
                for i, zone in enumerate(np.swapaxes(zone_pred, 0, 1)):
                    y_pred_zones[i].append(np.argmax(np.asarray(np.mean(zone, axis=0))))
                    try:
                        os.mkdir(f'{model_path}/{base_path}/chunks/zones')
                        os.mkdir(f'{model_path}/{base_path}/full_eeg/zones')
                    except Exception:
                        pass
                if combination == 'majority_voting':
                    max_pred_total = np.bincount(np.array(y_pred_zones)[:, -1]).argmax()
                else:
                    max_pred_total = np.mean(np.array(y_pred_zones), axis=1).argmax()
                y_pred.append(max_pred_total)
            else:
                y_pred.append(np.mean(pred, axis=0))
                # FULL EEGS FIRST
        if not zones or combination == 'mean':
            y_pred = np.argmax(np.asarray(y_pred), axis=1)

        cf_matrix = confusion_matrix(data[1], y_pred)

        with open(f'{model_path}/{base_path}/full_eeg/classification_report.txt',
                  'w') as f:
            print(classification_report(data[1], y_pred, labels=[0, 1], target_names=["No Parkinson", "Parkinson"]), file=f)
        print(cf_matrix)

        plot_confusion_matrix(cm=cf_matrix,
                              normalize=False,
                              target_names=["No Parkinson", "Parkinson"],
                              title="Matriz de confusión",
                              save=f'{model_path}/{base_path}/full_eeg/test_eeg.png')

        if y_pred_zones[0]:
            for i, y_pred in enumerate(y_pred_zones):
                cf_matrix = confusion_matrix(data[1], y_pred)
                print(cf_matrix)
                plot_confusion_matrix(cm=cf_matrix,
                                      normalize=False,
                                      target_names=["No Parkinson", "Parkinson"],
                                      title="Matriz de confusión",
                                      save=f'{model_path}/{base_path}/full_eeg/zones/test_confussion_zone_{i + 1}_matrix.png')

    # CHUNKS NOW

    if mode != 1:

        try:
            os.mkdir(f'{model_path}/{base_path}/chunks')
        except Exception:
            pass
        full, train, test, val = prepro.classification_generator_dataset()
        dataset, train_dataset, test_dataset, val_dataset = prepro.classification_tensorflow_dataset()
        if conjunto == 0 or conjunto == 4:
            _, y_data = zip(*list(test))
            data_dataset = test_dataset
        if conjunto == 1:
            _, y_data = zip(*list(train))
            data_dataset = train_dataset
        if conjunto == 2:
            _, y_data = zip(*list(val))
            data_dataset = val_dataset
        if conjunto == 3:
            _, y_data = zip(*list(full))
            data_dataset = dataset

        y_data = list(y_data)

        print("Chunks")

        y_pred = model.predict(data_dataset, verbose=1)
        y_pred_zones = []
        if len(y_pred) == 2:
            y_pred, y_pred_zones = y_pred
            y_pred_zones = np.argmax(np.swapaxes(y_pred_zones, 0, 1), axis=2)
        y_pred = np.argmax(y_pred, axis=1)

        cf_matrix = confusion_matrix(y_data, y_pred)

        with open(f'{model_path}/{base_path}/chunks/classification_report.txt',
                  'w') as f:
            print(classification_report(y_data, y_pred, labels=[0, 1], target_names=["No Parkinson", "Parkinson"]), file=f)
        print(cf_matrix)

        plot_confusion_matrix(cm=cf_matrix,
                              normalize=False,
                              target_names=["No Parkinson", "Parkinson"],
                              title="Matriz de confusión",
                              save=f'{model_path}/{base_path}/chunks/test_eeg.png')
        if y_pred_zones != []:
            for i, y_pred in enumerate(y_pred_zones):
                cf_matrix = confusion_matrix(y_data, y_pred)
                print(cf_matrix)
                plot_confusion_matrix(cm=cf_matrix,
                                      normalize=False,
                                      target_names=["No Parkinson", "Parkinson"],
                                      title="Matriz de confusión",
                                      save=f'{model_path}/{base_path}/chunks/zones/test_confussion_zone_{i + 1}_matrix.png')


if __name__ == "__main__":
    model_paths = [#"C:/Users/Ceiec01/OneDrive - UFV/PFG/Codigo/checkpoints/BERT-ControlesPre-CanalesReducidos",
                   #"C:/Users/Ceiec01/OneDrive - UFV/PFG/Codigo/checkpoints/BERT-HigherDropout-64",
                   "C:/Users/Ceiec01/OneDrive - UFV/PFG/Codigo/checkpoints/BERT-Zones-Final"
                   ]
    for model_path in model_paths:
        for conjunto in range(3):
            print(f'{model_path} - {conjunto}')
            test_model(model_path, conjunto, 1)
        print(f'{model_path} - post')
        test_model(model_path, 3, 2)
