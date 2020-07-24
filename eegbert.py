import argparse
import sys
import os

from main_func.inference import inference_wrapper, evaluation_wrapper
from main_func.training_batch import train_model
from config import *

main_desc = 'Este script es el punto de entrada de todos los modos de eeg-bert. Permite llevar a cabo entrenamiento' \
            ', inferencia y evaluación de los modelos.'


def main(arguments):
    parser = argparse.ArgumentParser(description=main_desc)

    mode_parser = parser.add_mutually_exclusive_group()
    mode_parser.add_argument('-t', '--training', action='store_true')
    mode_parser.add_argument('-i', '--inference', action='store_true')
    mode_parser.add_argument('-e', '--evaluation', action='store_true')
    # training
    parser.add_argument('-b', '--batch_size', default=16, type=int)

    parser.add_argument('--epochs', default=5, type=int)

    parser.add_argument('-m', '--modelo',
                        help='Modelo a entrenar. ', type=str, choices=choices_modelo.keys())
    parser.add_argument('--name',
                        help='Ruta y nombre del modelo. ', type=str)
    parser.add_argument('--nopad', action='store_true', help="Si se especifica el modelo no tendrá padding")
    parser.add_argument('--paciente',
                        help='Que tipo de pacientes se van a usar en el conjunto de pacientes. ', choices=choices_paciente.keys(),
                        default=next(iter(choices_paciente.keys())))
    parser.add_argument('--prueba', help='Prueba hecha a los pacientes, FTD, FTI, Resting o todos', choices=choices_prueba.keys())
    parser.add_argument('-c', '--config_file',
                        help='Ruta del archivo json que contiene los datos del modelo')

    parser.add_argument('--win_width', default=256, type=int, help='Anchura de la ventana')

    parser.add_argument('--win_step', default=1, type=int, help='Posiciones que se desplaza la ventana')

    # inference
    parser.add_argument('--path',
                        help='Expresión regular que hace match sobre los directorios de los modelos a probar.',
                        type=str, default=f'{os.getcwd()}')
    parser.add_argument('--dataset', help='Regex que matchee los archivos que se quieran clasificar', type=str)
    parser.add_argument('--out_file', help='Ruta del fichero de salida con el nombre', default="inference.csv", type=str)

    # evaluation
    parser.add_argument('--mode', help='full, chunks, both',
                        type=str, default=next(iter(choices_mode.keys())), choices=choices_mode.keys())
    parser.add_argument('--in_file', help='Ruta del fichero de entrada de la inferencia. Si no se especifica, asume que en el working directory'
                                          'hay una carpeta llamada inference con un fichero inference.csv en el', default='./inference/'
                                                                                               'inference.csv',type=str)
    parser.add_argument('--real_file', help='Ruta del archivo que contienen las predicciones. El formato es csv, la primera'
                                            'columna es la ruta del archivo completa y la segunda es la etiqueta.')
    parser.add_argument('--conjunto',
                        help='Indica un conjunto especifico a probar. test, train, val, full, test_pre_post',
                        type=str, default=next(iter(choices_conjunto.keys())), choices=choices_conjunto.keys())
    parser.add_argument('--combination',
                        help='mean o majority_voting para la combinación de los resultados en el modelo de zonas. ',
                        type=str, default='mean', choices=['mean', 'majority_voting'])
    parser.add_argument('--by_class',
                        help='Si se va a realizar la prueba para cada una de las clases',  action='store_true')

    args = parser.parse_args(arguments)
    modelo = choices_modelo[args.modelo] if args.modelo else None
    prueba = choices_prueba[args.prueba] if args.prueba else None
    epochs = args.epochs
    batch_size = args.batch_size
    win_width = args.win_width
    win_steps = args.win_step
    paciente = choices_paciente[args.paciente] if args.paciente else None
    nopad = args.nopad
    config_file = args.config_file
    path = args.path
    dataset = args.dataset
    out_file = args.out_file
    in_file = args.in_file
    conjunto = choices_conjunto[args.conjunto] if args.conjunto else None
    real_file = args.real_file
    combination = args.combination
    mode = choices_mode[args.mode] if args.mode else None
    by_class = args.by_class
    name = args.name

    if args.training:
        train_model(modelo, prueba, total_epoch_count=epochs, batch_size=batch_size,
                    window_width=win_width, window_steps=win_steps, patient=paciente,
                    nopad=nopad, json=config_file, name=name)
    if args.inference:
        inference_wrapper(path, dataset, out_file)
    if args.evaluation:
        evaluation_wrapper(in_file, paciente, conjunto, real_file, prueba, combination,
        mode, batch_size, by_class)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))