import glob
import json

from BERT import BertModelLayer, SplitterLayer
import tensorflow as tf
# print(tf.version)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def load_model(model_path):
    weights = glob.glob(model_path + "/training_weights/weights*.hdf5")
    with open(model_path + "\\model_architecture.json", 'r') as f:
        model_arch = json.load(f)
    model = tf.keras.models.model_from_json(model_arch, custom_objects={'BertModelLayer': BertModelLayer,
                                                                        'SplitterLayer': SplitterLayer})
    model.load_weights(weights[-1])
    return model
