from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import random
import sys
from tqdm.auto import tqdm
import easygui
import numpy as np

# Directorio base donde se encuentra el dataset
base_path = "C:\\Users\\Ceiec01\\OneDrive - UFV\\datasets\\EEGs_Pre_Post_LD"
version = "v1.0.0"

random.seed(42)

pruebas = {-1: "*",
           0: "FTD",
           1: "FTI",
           2: "Resting"}
limpios = {-1: "*",
           0: "EEGs_brutos",
           1: "EEgs_limpieza_CSIC"}


def refactor_data():
    """
    Function that collect all eeg binary files, extracts data and timestamps info
    and saves them to npy files in the same folder so that importing later is easier
    :return: Nothing
    """
    print("Building dataset from scratch...")
    path = os.path.join(base_path, version)
    # Semilla para reproducibilidad
    random.seed(42)

    control_path = os.path.join(path, 'controles')
    no_control_path = os.path.join(path, 'pacientes')

    control_files = glob.glob(control_path + '\\*\\*[!y]', recursive=True)
    no_control_files = glob.glob(no_control_path + '\\*\\*[!y]', recursive=True)

    for file in tqdm(control_files + no_control_files):
        print(file)
        f = open(file, "r")
        times = np.asarray([float(i) for i in f.readline().split(sep="\t")[1:-1]])
        # np.save(file+'_timestamp', times)
        eeg = np.empty((64, len(times)))
        for i in range(64):
            line = f.readline().split(sep="\t")
            eeg[i] = np.asarray([float(i) for i in line[1:-1]])
        # np.save(file+'_eeg', eeg)


def gui_temp(gui=False):
    """
    Placeholder function for a gui guided importing of files
    :param gui: Bool to show gui or not
    :return: Nothing
    """
    if gui:
        from_file = easygui.ynbox(msg='Cargar el dataset de nuevo?', title=' ',
                                  default_choice='[<F2>]No', cancel_choice='[<F2>]No')
        path = easygui.diropenbox(msg="Seleccione raiz del dataset", title="Seleccione raiz del dataset")

        limpio = easygui.indexbox(msg='Seleccione limpios o brutos',
                                  choices=([limpios[i] for i in limpios]),
                                  cancel_choice=limpios[0])
        prueba = easygui.indexbox(msg='Seleccione la prueba a cargar',
                                  choices=([pruebas[i] for i in pruebas]),
                                  cancel_choice=pruebas[0])
    else:
        from_file = True
        path = os.path.join(base_path, version)
        if not path:
            sys.exit(0)  # exit the program+
        limpio = -1
        prueba = -1

    control_path = os.path.join(path, 'controles')
    no_control_path = os.path.join(path, 'pacientes')

    # Cogemos los ficheros tanto de control como de no control
    try:
        control_files = glob.glob(control_path + '\\' + limpios[limpio] + '\\*_' + pruebas[prueba] + '[!y]',
                                  recursive=True)
        no_control_files = glob.glob(no_control_path + '\\' + limpios[limpio] + '\\*_' + pruebas[prueba] + '[!y]',
                                     recursive=True)
    except KeyError:
        control_files = glob.glob(control_path + '\\*\\*_*[!y]', recursive=True)
        no_control_files = glob.glob(no_control_path + '\\*\\*_*[!y]', recursive=True)


def load_data(limpio=-1, prueba=-1):
    """
    Function that gets all npy files for eeg
    :param limpio: select clean eeg (1), raw (0) or both (-1)
    :param prueba: select Right Hand Finger Taping (0), Left Hand Finger Taping (1), Resting (2) or all (-1)
    :return: Three lists, first for controls, second for patients pre-levo and third patients post-levo.
    """
    if limpio not in limpios.keys():
        limpio = -1
    if prueba not in pruebas.keys():
        prueba = -1

    path = os.path.join(base_path, version)

    control_path = os.path.join(path, 'controles')
    no_control_path = os.path.join(path, 'pacientes')
    control_files = glob.glob(control_path + '\\'+limpios[limpio]+'\\*_eeg.npy', recursive=True)
    no_control_files_pre = glob.glob(no_control_path + '\\'+limpios[limpio]+'\\*Pre_'+pruebas[prueba]+'_eeg.npy',
                                     recursive=True)
    no_control_files_post = glob.glob(no_control_path + '\\'+limpios[limpio]+'\\*Post_'+pruebas[prueba]+'_eeg.npy',
                                      recursive=True)

    control_array = []
    no_control_pre_array = []
    no_control_post_array = []

    for i, array in tqdm(enumerate(control_files + no_control_files_pre + no_control_files_post)):
        eeg = np.load(array)
        if i < len(control_files)-1:
            control_array.append(eeg)
        elif i < len(no_control_files_pre)-1:
            no_control_pre_array.append(eeg)
        else:
            no_control_post_array.append(eeg)

    return control_array, no_control_pre_array, no_control_post_array


def generate_dataset(exclude=None):
    """
    Function that returns a full dataset as a dictionary
    :param exclude: A list containing two list with keys to exclude. First list refers to limpios' keys and second
    to pruebas' keys
    :return: Nested dictionaries that allow selection of multiple different combinations
    """
    if not exclude:
        exclude = [[], []]
    if not isinstance(exclude, list):
        pass
    if len(exclude) == 2:
        if not isinstance(exclude[0], list):
            pass
        if not isinstance(exclude[1], list):
            pass

    dataset = {}
    for limpio in tqdm(limpios.keys()):
        if limpio in exclude[0]:
            continue
        prueba_dict = {}
        for prueba in tqdm(pruebas.keys()):
            if prueba in exclude[1]:
                continue
            control, pat_pre, pat_post = load_data(limpio, prueba)
            patiente_dict = {"control": control, "pat_pre": pat_pre, "pat_post": pat_post}
            prueba_dict[pruebas[prueba]] = patiente_dict
        dataset[limpios[limpio]] = prueba_dict

    return dataset


if __name__ == "__main__":
    generate_dataset()
