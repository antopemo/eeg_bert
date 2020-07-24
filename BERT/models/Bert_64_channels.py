from BERT import StockBertConfig, BertModelLayer
from BERT.loader import map_stock_config_to_params
from tensorflow import keras
import numpy as np
from aux_func.confussion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from aux_func.load_model import load_model
from config import bert_config_64
import json as json_app

def create_model(input_shape, adapter_size=64, json=None):
    """Creates a classification model."""

    # adapter_size = 64  # see - arXiv:1902.00751

    # create the bert layer

    if json:
        try:
            json_app.loads(json)
            bc = StockBertConfig.from_json_string(json)
        except ValueError:
            bc = StockBertConfig.from_json_file(json)
    else:
        bc = StockBertConfig.from_json_string(bert_config_64)

    bert_params = map_stock_config_to_params(bc)
    bert_params.adapter_size = adapter_size
    bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=input_shape, dtype='float32', name="input_ids")

    output = bert(input_ids)

    print("bert shape", output.shape)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=2, activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, input_shape))

    model.summary()

    return model


def test_model(model_path, prepro, model=None):
    if model is None:
        model = load_model(model_path)
    test = prepro.test_set
    y_pred = []
    for x_test, y_test in zip(test[0], test[1]):
        test_dataset = prepro.tf_from_generator([x_test], [y_test])
        pred = model.predict(test_dataset, verbose=1)
        y_pred.append(np.mean(pred, axis=0))

    y_pred = np.argmax(np.asarray(y_pred), axis=1)

    cf_matrix = confusion_matrix(test[1], y_pred)

    with open(model_path + '/classification_report.txt', 'w') as f:
        print(classification_report(test[1], y_pred, target_names=["No Parkinson", "Parkinson"]), file=f)
    print(cf_matrix)

    plot_confusion_matrix(cm=cf_matrix,
                          normalize=False,
                          target_names=["No Parkinson", "Parkinson"],
                          title="Matriz de confusión",
                          save=model_path + f'/test_eeg_postlevo.png')