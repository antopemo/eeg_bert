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

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def plot_model(model, model_path):
    tf.keras.utils.plot_model(
        model, to_file=model_path + '\\model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )


if __name__ == "__main__":
    path = "C:\\Users\\Ceiec01\\OneDrive - UFV\\PFG\\Codigo\\checkpoints\\BERT-Sigmoid-Final"
    plot_model(load_model(path), path)
