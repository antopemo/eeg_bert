import argparse
import glob
import os
import sys
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from aux_func.confussion_matrix import plot_confusion_matrix
from aux_func.data_preprocess import Preprocessor
from aux_func.load_model import load_model


def check_file_is_completed(inference_file_handler, inference_path):
    if not os.path.isfile(inference_path):
        return 0, None
    if os.path.isfile(inference_path):
        lines = inference_file_handler.read().splitlines()
        last_line = lines[-1]
        if last_line == 'END':
            return -1, lines[1:-1]
        else:
            return len(lines) - 1, lines


def write_inference_all(model, inference_path, batch_size=16):
    inference_file = f'{inference_path}\\inference.csv'

    channels = []
    if model.input_shape[2] == 25:
        channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]
    out_shape = list(model.input_shape[1:])
    prepro = Preprocessor(batch_size, model.input_shape[1], 64, prueba=-1, limpio=0, paciente=-1,
                          channels=channels, output_shape=out_shape)
    try:
        os.mkdir(inference_path)
    except Exception:
        pass

    with open(inference_file, 'w+') as out_file:
        line_from, file_content = check_file_is_completed(out_file, inference_path)

        if line_from == -1:
            return file_content

        out_file.write('START\n')

    data = prepro.dataset
    zones = False
    if len(model.output_shape) > 2:
        zones = True

    for x_data, y_data in zip(data[0][line_from:], data[1][line_from:]):
        test_dataset = prepro.tf_from_generator([x_data], [y_data])
        predictions = model.predict(test_dataset, verbose=1)
        with open(inference_file, 'a+') as out_file:
            for x, pred in enumerate(predictions):
                if zones:
                    for i, zone in enumerate(pred[1]):
                        no_park_pred = zone[0]
                        park_pred = zone[1]
                        out_file.write(f'{x_data};{x};{i};{no_park_pred};{park_pred}\n')
                else:
                    no_park_pred = pred[0]
                    park_pred = pred[1]
                    out_file.write(f'{x_data};{x};{-1};{no_park_pred};{park_pred}\n')

    with open(inference_file, 'a+') as out_file:
        out_file.write('END')
        return out_file.read().splitlines()[1:-1]


def inference(model_path, conjunto, patient, prueba=1, combination='mean', mode=0, batch_size=16):

    model = load_model(model_path)
    conjuntos = ['test', 'train', 'val', 'full', 'test_pre_post']
    patients = ['control', 'pre', 'post', 'Pre-Post']
    pruebas = {-1: "Both",
               0: "FTD",
               1: "FTI",
               2: "Resting"}
    inference_path = f'{model_path}\\inference'

    results = write_inference_all(model, inference_path)
    dataframe = pd.read_csv(StringIO('\n'.join(results)), names=['EEG_file', 'Chunk', 'Zona', 'No_Parkinson', 'Parkinson'],
                            sep=';')

    base_path = f'{conjuntos[conjunto]}_{patients[patient]}_{pruebas[prueba]}'
    zones = False
    if 'Zone' in model_path:
        base_path = f'{base_path}_{combination}'
        zones = True
    print(f'{base_path}')

    test_post = False
    if conjunto == 4:
        test_post = True

    prepro = Preprocessor(batch_size,
                          256,
                          64,
                          prueba=prueba,
                          limpio=0,
                          paciente=patient,
                          channels=[],
                          transpose=True,
                          test_post=test_post)

    if mode != 2:

        print("Full EEGs")
        try:
            os.mkdir(f'{model_path}\\{base_path}\\full_eeg')
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

        if not zones:
            full_eeg = dataframe['EEG_file', 'Zona', 'No_Parkinson', 'Parkinson'].groupby(['EEG_file', 'Zona']).mean()
            y_pred = full_eeg['No_Parkinson', 'Parkinson'].loc[full_eeg['EEG_file'].isin(data[0])].values
            y_pred = np.argmax(np.asarray(y_pred), axis=1)
        else:
            full_eeg = dataframe['EEG_file', 'Zona', 'No_Parkinson', 'Parkinson'].groupby(['EEG_file', 'Zona']).mean()
            y_pred = full_eeg['No_Parkinson', 'Parkinson'].loc[full_eeg['EEG_file'].isin(data[0])].values
            y_pred = np.argmax(np.asarray(y_pred), axis=1)
        cf_matrix = confusion_matrix(data[1], y_pred)
        with open(f'{model_path}\\{base_path}/full_eeg/classification_report.txt', 'w') as f:
            print(classification_report(data[1], y_pred, labels=[0, 1], target_names=["No Parkinson", "Parkinson"]), file=f)
        print(cf_matrix)

        plot_confusion_matrix(cm=cf_matrix,
                              normalize=False,
                              target_names=["No Parkinson", "Parkinson"],
                              title="Matriz de confusión",
                              save=f'{model_path}\\{base_path}\\full_eeg\\test_eeg.png')

def main(arguments):
    parser = argparse.ArgumentParser(
        description='Este script se usa para llevar a cabo predicciones sobre el dataset sobre el que se ha'
                    'entrenado un modelo.  Permite seleccionar sobre un set de modelos almacenados en un directorio'
                    'sobre cual se realizan las pruebas.')
    parser.add_argument('--path',
                        help='Expresión regular que hace match sobre los directorios de los modelos a probar.',
                        type=str, default=f'{os.getcwd()}')
    parser.add_argument('--modelo', help='Posición del modelo de entre que hacen match sobre path se va a probar. '
                                         'Por defecto el primero que matchea.', type=int, default=0)
    parser.add_argument('--conjunto',
                        help='Indica un conjunto especifico a probar. test, train, val, full, test_pre_post',
                        type=int, default=None)
    parser.add_argument('--combination',
                        help='mean o majority_voting para la combinación de los resultados en el modelo de zonas. ',
                        type=str, default='mean', choices=['mean', 'majority_voting'])
    parser.add_argument('--mode',
                        help='0: full y chunks, 1 full, 2 chunks',
                        type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--patient',
                        help='Que tipo de pacientes se van a usar en el conjunto de inferencia. '
                             '[Pre-post, controles, pre, post]', type=int, default=1, choices=[-1, 0, 1, 2])
    args = parser.parse_args(arguments)

    paths = glob.glob(args.path)
    model_path = paths[args.modelo]
    print(model_path)
    if model_path.find('Both') != -1:
        prueba = -1
    if model_path.find('FTI') != -1:
        prueba = 1
    if model_path.find('FTD') != -1:
        prueba = 0

    inference(model_path, args.conjunto, args.patient, prueba, args.combination, args.mode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
