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
    parser.add_argument('--conjunto', help='Indica un conjunto especifico a probar. test, train, val, full, test_pre_post', type=int, default=None)
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
    if not args.conjunto:
        for conjunto in range(4):
            print(f'{model_path} - {conjunto} - {prueba}')
            test_model(model_path, conjunto, args.patient, prueba, args.combination, args.mode)
        print(f'{model_path} - post- {prueba}')
        test_model(model_path, 3, 2, prueba, args.combination, args.mode)
    else:
        test_model(model_path, args.conjunto, args.patient, prueba, args.combination, args.mode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
