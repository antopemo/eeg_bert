import tensorflow as tf
from BERT import StockBertConfig, BertModelLayer
from BERT.loader import map_stock_config_to_params
from tensorflow import keras


class BERT_model(tf.keras.layers):
    def __init__(self, input_shape, adapter_size=64, bert_config=None):
        self.input_shape = input_shape
        if bert_config is None:
            self.bert_config = f'{{ \
                            "attention_probs_dropout_prob": 0.3, \
                            "hidden_act": "gelu", \
                            "hidden_dropout_prob": 0.3, \
                            "hidden_size": 64,\
                            "initializer_range": 0.02,\
                            "intermediate_size": 1536,\
                            "max_position_embeddings": 5120,\
                            "num_attention_heads": 4,\
                            "num_hidden_layers": 6,\
                            "type_vocab_size": 2,\
                            "vocab_size": 30522\
                            }}'

        super(BERT_model, self).__init()
        # create the bert layer
        bc = StockBertConfig.from_json_string(bert_config)
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        self.bert = BertModelLayer.from_params(bert_params, name="bert")

    def call(self, inputs):
        input_ids = keras.layers.Input(shape=self.input_shape, dtype='float32', name="input_ids")

        output = self.bert(input_ids)

        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)
        logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
        logits = keras.layers.Dropout(0.5)(logits)
        logits = keras.layers.Dense(units=2, activation="softmax")(logits)

        return logits

