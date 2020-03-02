from RMDL import RMDL_Image as RMDL
from aux_func.data_preprocess import Preprocessor

batch_size = 10
window_width = 256
window_steps = 64
channels = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]
out_shape = [window_width, len(channels), 1] if channels else [window_width, 64, 1]

if __name__ == "__main__":
    train, test, val = Preprocessor(batch_size,
                                    window_width,
                                    window_steps,
                                    prueba=0,
                                    limpio=0,
                                    paciente=1,
                                    channels=channels,
                                    transpose=True,
                                    output_shape=out_shape).classification_generator_dataset()

    x_train, y_train = zip(*list(train))
    x_test, y_test = zip(*list(test))
    x_val, y_val = zip(*list(val))

    x_train = list(x_train)
    y_train = list(y_train)
    x_test = list(x_test)
    y_test = list(y_test)
    x_val = list(x_val)
    y_val = list(y_val)

    number_of_classes = 2
    sparse_categorical = 0

    n_epochs = [100, 100, 100]  # DNN--RNN-CNN
    Random_Deep = [0, 0, 3]  # DNN--RNN-CNN

    RMDL.Image_Classification(x_train, y_train, x_test, y_test, tuple(out_shape),
                              batch_size=batch_size,
                              sparse_categorical=True,
                              random_deep=Random_Deep,
                              epochs=n_epochs)
