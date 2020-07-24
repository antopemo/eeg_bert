# Rutas para el dataset y dentro las versiones. Hay que respetar la estructura de
# carpetas de la v1.0.0
dataset_base_path = "C:/Users/Ceiec01/OneDrive - UFV/datasets/EEGs_Pre_Post_LD"
dataset_version = "v2.0.0"

# Diferentes configuraciones de canales
channels_64 = []
channels_25 = [9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53]
channels_40 = [4, 5, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 38, 39,
               40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 54, 58, 59, 60]
channels_config = [channels_64, channels_25, channels_40]

bert_config_64 = f'{{ \
    "attention_probs_dropout_prob": 0.3, \
    "hidden_act": "gelu", \
    "hidden_dropout_prob": 0.3, \
    "hidden_size": 40,\
    "initializer_range": 0.02,\
    "intermediate_size": 1536,\
    "max_position_embeddings": 5120,\
    "num_attention_heads": 4,\
    "num_hidden_layers": 6,\
    "type_vocab_size": 2,\
    "vocab_size": 30522\
    }}'

bert_config_40 = f'{{ \
    "attention_probs_dropout_prob": 0.3, \
    "hidden_act": "gelu", \
    "hidden_dropout_prob": 0.3, \
    "hidden_size": 40,\
    "initializer_range": 0.02,\
    "intermediate_size": 1536,\
    "max_position_embeddings": 5120,\
    "num_attention_heads": 4,\
    "num_hidden_layers": 6,\
    "type_vocab_size": 2,\
    "vocab_size": 30522\
    }}'

bert_config_25 = f'{{ \
    "attention_probs_dropout_prob": 0.3, \
    "hidden_act": "gelu", \
    "hidden_dropout_prob": 0.3, \
    "hidden_size": 25,\
    "initializer_range": 0.02,\
    "intermediate_size": 1536,\
    "max_position_embeddings": 5120,\
    "num_attention_heads": 5,\
    "num_hidden_layers": 6,\
    "type_vocab_size": 2,\
    "vocab_size": 30522\
    }}'

#Se omite el 2 porque corresponde al de zonas

bert_configs = {0: (bert_config_64, channels_64),
                1: (bert_config_25, channels_25),
                3: (bert_config_40, channels_40)}
###########################################################################
#Estas variables son strs para los menús y los nombres de modelos
choices_modelo = {'64_ch': 0, '25_ch': 1, 'zones': 2, '40_ch': 3}
choices_paciente = {'pre_post': -1, 'control': 0, 'pre': 1, 'post': 2}
choices_prueba = {'FTD': 0, 'FTI': 1, "all": 2, 'resting': -1}
choices_mode = {'both': 0, 'full': 1, 'chunks': 2}
choices_conjunto = {'test': 0, 'train': 1, 'val': 2, 'full': 3, 'test_pre_post': 4}
choices_limpio = {"both": -1, "brutos": 0, "limpios": 1}


