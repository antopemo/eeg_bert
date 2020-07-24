from .data_reading import generate_dataset, load_and_slice
from sklearn.model_selection import train_test_split
import tensorflow as tf
import config


class Preprocessor:
    # Todo: Selection function to get from a file or config from within the model the params liked.

    def __init__(self,
                 batch_size,
                 window_width,
                 window_steps,
                 prueba,
                 limpio,
                 paciente,
                 channels,
                 test_size=0.2,
                 val_size=0.2,
                 control=0,
                 shuffle=True,
                 transpose=True,
                 output_shape=None,
                 test_post=False):
        """
        This function serves as a configurator for a certain preprocessor,
        allowing for multiple configurations to be made

        :param batch_size: Batch size of the datasets
        :param window_width: Width of the window to be sliced
        :param window_steps: How many points will the window be moved to take new data
        :param prueba: Which test will be taken into consideration
        :param limpio: If EEGs will be cleaned or raw
        :param paciente: Which of the different patient types is selected
        :param channels: Which channels to select
        :param test_size: Percentage of the total number of data to be used as test
        :param val_size: Percentage of the train set to be used as validation
        :param control: Which to select as controls
        :param shuffle: Whether or not to shuffle the dataset
        :param transpose: If True, the result will be time x channel, else channel x time
        :param output_shape: shape of the output
        :param test_post: Returns the train test as train plus post
        """
        # Zona de declaración de constantes
        self.CHANNELS = 64
        self.WINDOW_WIDTH = window_width
        self.WINDOW_STEPS = window_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transpose = transpose
        self.channels = channels
        self.dataset = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.test_post = test_post

        if output_shape is None:
            self.output_shape = [self.CHANNELS, self.WINDOW_WIDTH, 1]
        else:
            # assert output_shape == [self.CHANNELS, self.WINDOW_WIDTH, 1] or output_shape == [
            #     self.CHANNELS * self.WINDOW_WIDTH, 1] or output_shape == [
            #     self.CHANNELS * self.WINDOW_WIDTH]
            self.output_shape = output_shape

        pruebas = {v: k for k, v in config.choices_prueba.items()}
        limpios = {v: k for k, v in config.choices_limpio.items()}
        pacient = {v: k for k, v in config.choices_paciente.items()}

        # Cargamos el dataset. Esto devuelve un diccionario que podemos emplear para seleccionar qué tipos de datos
        # queremos usar. Quitamos las opciones de que cojan datasets mixtos.
        dataset = generate_dataset()
        if paciente != -1:
            self.patient = dataset[limpios[limpio]][pruebas[prueba]][pacient[paciente]]
        else:
            self.patient = dataset[limpios[limpio]][pruebas[prueba]][pacient[1]] +\
                           dataset[limpios[limpio]][pruebas[prueba]][pacient[2]]

        self.controles = dataset[limpios[limpio]][pruebas[prueba]][pacient[control]]
        self.post = dataset[limpios[limpio]][pruebas[prueba]][pacient[2]]

        self._split_dataset(test_size, val_size)

    def classification_tensorflow_dataset(self):
        """
        Function for returning a dataset in form of tf.dataset

        :return: four datasets: full, train, test, validation
        """
        (x, y), (x_train, y_train), (x_test, y_test), (x_val, y_val) = self._split_dataset()
        ds = self.tf_from_generator(x, y)

        ds_train = self.tf_from_generator(x_train, y_train)
        ds_test = self.tf_from_generator(x_test, y_test)
        ds_val = self.tf_from_generator(x_val, y_val)
        return ds, ds_train, ds_test, ds_val

    def classification_generator_dataset(self):
        """
        Function to return a dataset built as a generator

        :return: The full dataset
        """
        dataset = self._split_dataset()
        ds = []
        for x, y in dataset:
            ds.append(load_and_slice(x, y, self.WINDOW_STEPS,
                                     self.WINDOW_WIDTH,
                                     self.output_shape,
                                     self.transpose,
                                     self.channels))

        return ds

    def _split_dataset(self, test_size=0.2, val_size=0.2):
        """

        :param test_size: [0-1] Percentage of total data required for test
        :param val_size: [0-1] Percentage of train used for validation
        :return: 4 datasets: full, train, test and validation
        """
        if self.dataset:
            return self.dataset, self.train_set, self.test_set, self.val_set

        x = self.patient + self.controles
        y = [1] * len(self.patient) + [0] * len(self.controles)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        assert len(x_train) > len(x_test), 'Test length higher than train'

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)
        assert len(x_train) > len(x_val), 'Val length higher than train'

        self.dataset = (x, y)
        self.train_set = (x_train, y_train)
        self.test_set = (x_test, y_test)
        if self.test_post:
            x_post = self.post
            y_post = [1]*len(x_post)
            self.test_set = (x_test + x_post, y_test + y_post)
        self.val_set = (x_val, y_val)

        return self.dataset, self.train_set, self.test_set, self.val_set

    def tf_from_generator(self, x, y):
        """

        :param x: Input data paths
        :param y: Target data labels
        :return: Tf.Dataset with selected data
        """
        _data = tf.data.Dataset.from_generator(
            load_and_slice,
            (tf.float32, tf.int32),
            # When only passing int number , shape must be empty
            (tf.TensorShape(self.output_shape), tf.TensorShape([])),
            args=(x, y, self.WINDOW_STEPS, self.WINDOW_WIDTH, self.output_shape, self.transpose, self.channels))

        if self.shuffle:
            _data.shuffle(100000)
        # TODO: Podria hacerse un repeat de data para aumentar el dataset (??)
        return _data.batch(self.batch_size)
