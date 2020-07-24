from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import random
from tqdm.auto import tqdm
import numpy as np

# Directorio base donde se encuentra el dataset
import config

base_path = config.dataset_base_path
version = config.dataset_version

random.seed(42)

pruebas = {v: k for k, v in config.choices_prueba.items()}
limpios = {v: k for k, v in config.choices_limpio.items()}

pruebas_regex = {-1: "*",
           0: "FTD",
           1: "FTI",
           2: "Resting"}
limpios_regex = {-1: "*",
           0: "EEGs_brutos",
           1: "EEGs_limpieza_CSIC"}


def refactor_data(path=None):
    """
    Function that collect all eeg binary files, extracts data and timestamps info
    and saves them to npy files in the same folder so that importing later is easier
    :return: Nothing
    """
    print("Building dataset from scratch...")
    if not path:
        path = os.path.join(base_path, version)
    # Semilla para reproducibilidad
    random.seed(42)

    control_path = os.path.join(path, 'controles')
    no_control_path = os.path.join(path, 'pacientes')

    control_files = glob.glob(control_path + '/*/*[!y]', recursive=True)
    no_control_files = glob.glob(no_control_path + '/*/*[!y]', recursive=True)

    for file in tqdm(control_files + no_control_files):
        print(file)
        f = open(file, "r")
        times = np.asarray([float(i) for i in f.readline().split(sep="\t")[1:-1]])
        np.save(file + '_timestamp', times)
        eeg = np.empty((64, len(times)))
        for i in range(64):
            line = f.readline().split(sep="\t")
            eeg[i] = np.asarray([float(i) for i in line[1:-1]])
        np.save(file + '_eeg', eeg)


def load_data(limpio=-1, prueba=-1, path=None):
    """
    Function that gets all file's path for eeg
    :param limpio: select clean eeg (1), raw (0) or both (-1)
    :param prueba: select Right Hand Finger Taping (0), Left Hand Finger Taping (1), Resting (2) or all (-1)
    :return: Three lists, first for controls, second for patients pre-levo and third patients post-levo.
    """
    if limpio not in limpios.keys():
        limpio = -1
    if prueba not in pruebas.keys():
        prueba = -1
    if not path:
        path = os.path.join(base_path, version)

    control_path = os.path.join(path, 'controles')
    no_control_path = os.path.join(path, 'pacientes')
    control_files = glob.glob(control_path + '/' + limpios_regex[limpio] + '/*_' + pruebas_regex[prueba] + '_eeg.npy',
                              recursive=True)
    no_control_files_pre = glob.glob(
        no_control_path + '/' + limpios_regex[limpio] + '/*Pre_' + pruebas_regex[prueba] + '_eeg.npy',
        recursive=True)
    no_control_files_post = glob.glob(
        no_control_path + '/' + limpios_regex[limpio] + '/*Post_' + pruebas_regex[prueba] + '_eeg.npy',
        recursive=True)

    control_array = []
    no_control_pre_array = []
    no_control_post_array = []

    for i, array_path in tqdm(enumerate(control_files + no_control_files_pre + no_control_files_post)):

        # eeg = np.load(array)
        if i < len(control_files):
            control_array.append(array_path)
        elif i < len(control_files) + len(no_control_files_pre):
            no_control_pre_array.append(array_path)
        else:
            no_control_post_array.append(array_path)

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

    temp_dataset = {}
    for limpio in tqdm(limpios.keys()):
        if limpio in exclude[0]:
            continue
        prueba_dict = {}
        for prueba in tqdm(pruebas.keys()):
            if prueba in exclude[1]:
                continue
            control, pat_pre, pat_post = load_data(limpio, prueba)
            patiente_dict = {"control": control, "pre": pat_pre, "post": pat_post}
            prueba_dict[pruebas[prueba]] = patiente_dict
        temp_dataset[limpios[limpio]] = prueba_dict

    return temp_dataset


def pad_up_to(t, max_in_dims, constant_values):
    s = np.shape(t)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
    return np.pad(t, paddings, 'constant', constant_values=constant_values)


def load_and_slice(array_path, tag, step=16, width=64, output_shape=None, transpose=False, channels=None):
    """
    Generator to get data out of paths already sliced.
    :param array_path: Array of paths to get data from.
    :param tag: Array of tags. Must be the same length of array_path and have the same order.
    :param step: How many data points will the window slide.
    :param width: Width of the sliced portion.
    :param output_shape: Optional param to get the data out in a certain shape.
    :param transpose: transposes the matrix of the eeg for different training efforts
    :param channels: array of channels to get from source
    :return: a tuple containing one of the portions and its tag.
    """
    if channels is None:
        channels = []
    for i, path in enumerate(array_path):
        eeg = np.load(path)
        if channels != []:
            eeg = eeg[np.asarray(channels) - 1]
        for element in sliding_window(eeg, step, width):
            if transpose:
                element = element.T
            if output_shape is not None:
                # print(element.shape)
                shape_a, shape_b = 1, 1
                for a in output_shape:
                    shape_a *= a
                for b in element.shape:
                    shape_b *= b
                if shape_a == shape_b:
                    element = element.reshape(output_shape)
                else:
                    element = pad_up_to(element, output_shape, 0)
            yield element, tag[i]


def sliding_window(eeg, step=16, width=64):
    """
    Generator function that slices the EEG into smaller pieces of (64x Width).
    :param eeg: Array containing the different eeg readings to be sliced.
    :param step: How many data points will the window slide.
    :param width: Width of the sliding window.
    :return: List with the sliced arrays.
    """
    i = 0
    while i + step + width < eeg.shape[1]:
        chunk = eeg[:, i + step:i + step + width]
        i += 1
        # print(f'{i * step}: {eeg.shape} - {chunk.shape} - from {i*step} to {i*step+width}')

        yield chunk


if __name__ == "__main__":
    refactor_data()
