import os
import tensorflow as tf

from Common import Common
from Config import Config
from NaiveExtractor import NaiveExtractor
from Model import Model


SHOW_TOP_CONTEXTS = 10

class DataProcess(object):
    def __init__(self, model_path="../../data/models/java14_model/saved_model_iter8.release", config=None):
        self._config = config if config is not None else Config.get_default_config(load_path=model_path)
        self._path_extractor = NaiveExtractor(self._config)
        self._model = Model(self._config)

    def predict(self, file_path):
        try:
            predict_lines, hash_to_string_dict = self._path_extractor.extract_paths(file_path)
        except ValueError as e:
            print(e)
            return None
        results = self._model.predict(predict_lines)
        prediction_results = Common.parse_results(results, hash_to_string_dict, topk=SHOW_TOP_CONTEXTS)
        return prediction_results

    def get_path_feature_vec(self, org_path, is_evaluating=False):
        # 输入形式为：  int,(PrimitiveType0)^(MethodDeclaration)_(NameExpr1),METHOD_NAME

        split_path = org_path.split(',')
        src_word = split_path[0]
        str_path = split_path[1]
        tgt_word = split_path[2]




        with tf.variable_scope('model', reuse=self._model.get_should_reuse_variables()):
            # 获取words_vocab
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self._model.word_vocab_size + 1, self._model.config.EMBEDDINGS_SIZE),
                                      dtype=tf.float32, trainable=False)
            # 获取paths_vocab
            paths_vocab = tf.get_variable('PATHS_VOCAB',
                                      shape=(self._model.path_vocab_size + 1, self._model.config.EMBEDDINGS_SIZE),
                                      dtype=tf.float32, trainable=False)


