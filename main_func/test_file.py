"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Test file
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Este script se usa para probar una serie de archivos de prueba sobre los modelos.

Tiene tres parámetros:
    -modelos: Ruta escrita en REGEX que matchee las carpetas que contienen los modelos que queremos probar
            También se puede proporcionar un número, en cuyo caso se eligirá ese modelo de la carpeta predeterminada.
    -archivos: Ruta escrita en REGEX que matchee los ficheros a probar
    -out_file: Ruta de la salida csv o nombre de la misma.

"""

import argparse
import glob
import os
import sys
import re
import datetime

import numpy as np

from aux_func.data_preprocess import Preprocessor
from aux_func.data_reading import prepare_file
from aux_func.load_model import load_model


def test_model(model_path, file_path):
    model = load_model(model_path)
    file = prepare_file(file_path)
    channels = []
    if model.input_shape[2] == 25:
        channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]
    prepro = Preprocessor(16,
                          model.input_shape[1],
                          64,
                          prueba=0,
                          limpio=0,
                          paciente=0,
                          channels=channels,
                          transpose=True,
                          output_shape=list(model.input_shape[1:]))
    test_dataset = prepro.tf_from_generator([file], [0])
    pred = model.predict(test_dataset, verbose=0)
    if len(pred) == 2:
        pred, zone_pred = pred
        for i, zone in enumerate(np.swapaxes(zone_pred, 0, 1)):
            print(f'Predicción de la Zona {i}: {np.asarray(np.mean(zone, axis=0))}')
    # FULL EEGS FIRST
    print(f'\n\n\nPredicción total: {np.asarray((np.mean(pred, axis=0)))}')
    print(f'{"No Párkinson" if np.argmax([np.asarray((np.mean(pred, axis=0)))], axis=1) == [0] else "Párkinson"}')
    return np.asarray((np.mean(pred, axis=0)))





def main(arguments):
    parser = argparse.ArgumentParser('Este script se usa para probar una serie de archivos de prueba sobre los modelos')
    parser.add_argument('--modelos', help='Regex que matchee la ruta de los modelos', type=str)
    parser.add_argument('--archivos', help='Regex que matchee la ruta de los archivos', type=str)
    parser.add_argument('--out_file', help='Ruta del fichero de salida con el nombre', type=str)
    args = parser.parse_args(arguments)

    pattern = re.compile("\d+")

    if not args.modelos:
        path = 'C:/Users/Ceiec01/OneDrive - UFV/PFG/Codigo/checkpoints/Pruebas-paper/*'
        model_paths = glob.glob(path)
    elif pattern.fullmatch(args.modelos):
        path = 'C:/Users/Ceiec01/OneDrive - UFV/PFG/Codigo/checkpoints/Pruebas-paper/*'
        model = int(args.modelos)
        model_paths = [glob.glob(path)[model]]
    else:
        path = args.modelos
        model_paths = glob.glob(path)

    file_paths = glob.glob(args.archivos)


    for model_path in model_paths:
        for file_path in file_paths:
            model_name = model_path.split('/')[-1]
            file_name = file_path.split('/')[-1]
            print(file_name)
            print()
            test = test_model(model_path, file_path)
            with open(f'{args.out_file}_{file_name}.csv', 'a') as the_file:
                if os.path.getsize(f'{args.out_file}_{file_name}.csv') == 0:
                    the_file.write('Fichero;Modelo;NoParkinson;Parkinson\n')
                the_file.write(f'{file_name};{model_name};{test[0]};{test[1]}\n')


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
