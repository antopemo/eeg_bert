"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Predict All
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Este script se usa para llevar a cabo predicciones sobre el dataset sobre el que se ha
entrenado un modelo.  Permite seleccionar sobre un set de modelos almacenados en un directorio
sobre cual se realizan las pruebas.

Tiene dos parámetros:
    -path: Expresión regular que hace match sobre los directorios de los modelos a probar.
    -modelo: Posición del modelo de entre que hacen match sobre path se va a probar. Por defecto el primero que matchea.
"""

import argparse
import glob
import sys

from predictions import test_model


def main(arguments):
    parser = argparse.ArgumentParser(
        description='Este script se usa para llevar a cabo predicciones sobre el dataset sobre el que se ha'
                    'entrenado un modelo.  Permite seleccionar sobre un set de modelos almacenados en un directorio'
                    'sobre cual se realizan las pruebas.')
    parser.add_argument('--path',
                        help='Expresión regular que hace match sobre los directorios de los modelos a probar.',
                        type=str)
    parser.add_argument('--modelo', help='Posición del modelo de entre que hacen match sobre path se va a probar. '
                                         'Por defecto el primero que matchea.', type=int, default=0)
    parser.add_argument('--run', help='Indica una run especifica a probar. ', type=int, default=None)
    parser.add_argument('--combination',
                        help='mean o majority_voting para la combinación de los resultados en el modelo de zonas. ',
                        type=str, default='mean', choices=['mean', 'majority_voting'])
    args = parser.parse_args(arguments)
    if not args.path:
        path = 'C:\\Users\\Ceiec01\\OneDrive - UFV\\PFG\\Codigo\\checkpoints\\Pruebas-paper\\*'
    else:
        path = args.path

    paths = glob.glob(path)
    model_path = paths[args.modelo]
    print(model_path)
    if model_path.find('Both') != -1:
        prueba = -1
    if model_path.find('FTI') != -1:
        prueba = 1
    if model_path.find('FTD') != -1:
        prueba = 0
    if not args.run:
        for run in range(3):
            print(f'{model_path} - {run} - {prueba}')
            test_model(model_path, run, 1, prueba, args.combination)
        print(f'{model_path} - post- {prueba}')
        test_model(model_path, 3, 2, prueba, args.combination)
    else:
        test_model(model_path, args.run, 1, prueba, args.combination)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
