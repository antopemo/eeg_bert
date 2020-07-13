import datetime
import os
import numpy as np
import json

from sklearn.metrics import confusion_matrix, classification_report

from BERT import StockBertConfig, BertModelLayer
from BERT.loader import map_stock_config_to_params
import tensorflow as tf
from tensorflow import keras

from BERT.splitter_layer import SplitterLayer
from aux_func.confussion_matrix import plot_confusion_matrix
from aux_func.data_preprocess import Preprocessor

batch_size = 16
window_width = 256
window_steps = 64
channels = []

bert_config_file = os.path.join("BERT", "bert_config.json")

learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
train_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="SparseCatAc")

out_shape = [window_width, 64]

bert_config_base = f'{{ \
    "attention_probs_dropout_prob": 0.3, \
    "hidden_act": "gelu", \
    "hidden_dropout_prob": 0.5, \
    "initializer_range": 0.02,\
    "intermediate_size": 256,\
    "max_position_embeddings": 5120,\
    "num_hidden_layers": 6,\
    "type_vocab_size": 2,\
    "vocab_size": 30522\
    }}'
bert_config_zone = [f'{{"num_attention_heads": 7, "hidden_size": 14,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 4, "hidden_size": 16,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 5, "hidden_size": 10,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 3, "hidden_size": 6,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 5, "hidden_size": 5,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 6, "hidden_size": 12,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 4, "hidden_size": 16,' + bert_config_base[1:],
                    f'{{"num_attention_heads": 7, "hidden_size": 14,' + bert_config_base[1:]]


def create_model(input_shape, adapter_size=64):
    """Creates a classification model."""

    # adapter_size = 64  # see - arXiv:1902.00751

    # create the bert layer
    if np.ndim(input_shape) > 2:
        input_shape = np.reshape(input_shape, input_shape[:-1])
    bert = []
    for i, bert_config in enumerate(bert_config_zone):
        bc = StockBertConfig.from_json_string(bert_config)
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert.append(BertModelLayer.from_params(bert_params, name="bert_zone_" + str(i)))

    input_ids = keras.layers.Input(shape=input_shape, dtype='float32', name="input_ids")
    split_input = SplitterLayer(name="splitter")(input_ids)

    output = []
    cls_out = []
    logits = []
    for input, bert_model in zip(split_input, bert):
        output.append(bert_model(input))

        cls_out.append(keras.layers.Lambda(lambda seq: seq[:, 0, :])(output[-1]))
        cls_out[-1] = keras.layers.Dropout(0.5)(cls_out[-1])
        logits.append(keras.layers.Dense(units=768, activation="tanh")(cls_out[-1]))
        logits[-1] = keras.layers.Dropout(0.5)(logits[-1])
        logits[-1] = keras.layers.Dense(units=2, activation="softmax")(logits[-1])

    zone_model = []
    for zone in range(8):
        zone_model.append(keras.Model(inputs=input_ids, outputs=logits[zone]))
        zone_model[-1].build(input_shape=(None, input_shape))
    logits = tf.stack(logits, 0)
    logits = tf.transpose(logits, perm=[1, 0, 2])

    final_logits = tf.reduce_mean(logits, axis=1)

    model = keras.Model(inputs=input_ids, outputs=[final_logits, logits])
    model.build(input_shape=(None, input_shape))

    model.summary()

    return model, zone_model