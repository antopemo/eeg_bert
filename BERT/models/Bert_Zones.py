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


def train_test():
    prepro = Preprocessor(batch_size,
                          window_width,
                          window_steps,
                          prueba=0,
                          limpio=0,
                          paciente=1,
                          channels=channels,
                          transpose=True,
                          output_shape=out_shape
                          )
    train, test, val = prepro.classification_generator_dataset()

    _, y_train = zip(*list(train))
    _, y_test = zip(*list(test))
    _, y_val = zip(*list(val))

    y_train = list(y_train)
    y_test = list(y_test)
    y_val = list(y_val)

    dataset, train_dataset, test_dataset, val_dataset = prepro.classification_tensorflow_dataset()

    model = create_model(tuple(out_shape), adapter_size=None)
    model.compile(optimizer=optimizer,
                  loss=[train_loss, None],
                  metrics=[train_accuracy, 'acc'])

    model_path = os.path.normpath(
        "./checkpoints/BERT-Zones" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(model_path)
    os.mkdir(model_path + '\\training_weights')

    checkpoint_path = os.path.join(model_path, "training_weights\\weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    with open(model_path + '\\model_architecture.json', 'w') as f:
        json.dump(model.to_json(), f)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_freq='epoch',
                                                     verbose=1)

    log_dir = ".log\\eegs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1,
                                                       write_graph=True,
                                                       write_images=True,
                                                       update_freq='epoch',
                                                       profile_batch=2)

    total_epoch_count = 10
    model.fit(x=train_dataset,
              validation_data=val_dataset,
              shuffle=True,
              epochs=total_epoch_count,
              callbacks=[keras.callbacks.EarlyStopping(monitor="SparseCatAc", patience=10, restore_best_weights=True),
                         tensorboard_callback, cp_callback])

    os.mkdir(model_path + '\\test')
    os.mkdir(model_path + '\\test\\chunks')
    os.mkdir(model_path + '\\test\\full_eeg')

    test = prepro.test_set
    y_pred = []
    y_pred_zones = [[] for i in range(8)]
    for x_test, y_test in zip(test[0], test[1]):
        test_dataset = prepro.tf_from_generator([x_test], [y_test])
        pred = model.predict(test_dataset, verbose=1)
        y_pred.append(np.mean(pred[0], axis=0))
        for i, zone in enumerate(np.swapaxes(pred[1], 0, 1)):
            y_pred_zones[i].append(np.mean(zone, axis=0))
    # FULL EEGS FIRST
    y_pred = np.argmax(np.asarray(y_pred), axis=1)

    cf_matrix = confusion_matrix(test[1], y_pred)

    with open(model_path + '/test/full_eeg/classification_report.txt', 'w') as f:
        print(classification_report(test[1], y_pred, target_names=["No Parkinson", "Parkinson"]), file=f)
    print(cf_matrix)

    plot_confusion_matrix(cm=cf_matrix,
                          normalize=False,
                          target_names=["No Parkinson", "Parkinson"],
                          title="Matriz de confusión",
                          save=model_path + f'\\test\\full_eeg\\test_eeg.png')

    # CHUNKS NOW
    y_pred = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
    plot_confusion_matrix(cm=cf_matrix,
                          normalize=False,
                          target_names=["No Parkinson", "Parkinson"],
                          title="Matriz de confusión",
                          save=model_path + f'\\test\\chunks\\test_confussion_matrix.png')
