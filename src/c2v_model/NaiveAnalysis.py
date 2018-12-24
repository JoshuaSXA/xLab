import tensorflow as tf
import torch

from Common import Common
from Config import Config
from NaiveExtractor import NaiveExtractor
from Model import Model
import PathContextReader

SHOW_TOP_CONTEXTS = 10

EMBEDDING_DIM = 128


class PathAnalysis(object):
    def __init__(self, model_path="../../data/models/java14_model/saved_model_iter8.release", config=None):
        self._config = config if config is not None else Config.get_default_config(load_path=model_path)
        self._path_extractor = NaiveExtractor(self._config)
        self._model = Model(self._config)
        # 从Code2Vec模型中加载embedding matrix
        with tf.variable_scope('model', reuse=None):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self._model.word_vocab_size + 1, self._model.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)
            paths_vocab = tf.get_variable('PATHS_VOCAB',
                                           shape=(self._model.path_vocab_size + 1, self._model.config.EMBEDDINGS_SIZE),
                                           dtype=tf.float32, trainable=False)
            attention_param = tf.get_variable('ATTENTION',
                                               shape=(self._model.config.EMBEDDINGS_SIZE * 3, 1), dtype=tf.float32)
            transform_param = tf.get_variable('TRANSFORM',
                                              shape=(self._model.config.EMBEDDINGS_SIZE * 3, self._model.config.EMBEDDINGS_SIZE * 3),
                                              dtype=tf.float32)
            self._model.initialize_session_variables(self._model.sess)
            self._model.saver = tf.train.Saver()
            self._model.load_model(self._model.sess)
            self.words_vocab, self.paths_vocab, self.attention_param, self.transform_param  = self._model.sess.run([words_vocab, paths_vocab, attention_param, transform_param])
        # AST树的节点
        self._tree_root = ''
        # AST树的节点set
        self._node_set = None
        # AST节点和特征向量的dict
        self._node_vec_dict = {}

    # 计算每个节点对应的特征向量
    def predict_file(self, path_to_file):
        # 重置参数
        self._tree_root = ''
        self._node_set = None
        self._node_vec_dict = {}
        try:
            self._path_extractor.extract_paths(path_to_file)
        except ValueError as e:
            return None
        # 获取AST的根节点的hash_code
        self.get_tree_root()
        # 获取路径的特征向量和注意力
        paths = self._path_extractor.get_paths_list()
        context_embed_vec, attention_weights = self.path_parser(paths)
        # 遍历整个AST的节点，计算特征向量
        path_nodes = self._path_extractor.get_path_hash()
        try:
            for node in self._node_set:
                feature_vec = torch.FloatTensor(torch.zeros(EMBEDDING_DIM * 3, 1))
                for i in range(0, len(path_nodes)):
                    if node in path_nodes[i]:
                        # 如果该节点在该path中则就要考虑该节点的特征和注意力
                        embed_vec = context_embed_vec[i]
                        attention = attention_weights[i].view(1, 1)
                        tmp_feature = embed_vec.mm(attention)
                        feature_vec = feature_vec + tmp_feature
                if torch.sum(feature_vec) != 0.0:
                    feature_vec = (feature_vec - torch.min(feature_vec)) / (torch.max(feature_vec) - torch.min(feature_vec))
                self._node_vec_dict[node] = feature_vec
        except:
            raise ValueError

    def line_parser(self, org_path, is_evaluating=False):

        split_path = org_path.split(',')
        source_word = split_path[0]
        hash_path = str(NaiveExtractor.java_string_hashcode(split_path[1]))
        target_word = split_path[2]

        try:
            source_input = self._model.word_to_index[source_word]
            source_word_embed = torch.FloatTensor(self.words_vocab[source_input]).view(EMBEDDING_DIM, 1)
        except:
            source_word_embed = torch.FloatTensor(torch.zeros(EMBEDDING_DIM, 1))

        try:
            path_input = self._model.path_to_index[hash_path]
            path_embed = torch.FloatTensor(self.paths_vocab[path_input]).view(EMBEDDING_DIM, 1)
        except:
            path_embed = torch.FloatTensor(torch.zeros(EMBEDDING_DIM, 1))

        try:
            target_input = self._model.word_to_index[target_word]
            target_word_embed = torch.FloatTensor(self.words_vocab[target_input]).view(EMBEDDING_DIM, 1)
        except:
            target_word_embed = torch.FloatTensor(torch.zeros(EMBEDDING_DIM, 1))

        # 转为tensor
        transform_param = torch.FloatTensor(self.transform_param).view(EMBEDDING_DIM * 3, EMBEDDING_DIM * 3)
        attention_param = torch.FloatTensor(self.attention_param).view(EMBEDDING_DIM * 3, 1)
        context_embed = torch.cat([source_word_embed, path_embed, target_word_embed], dim=0)

        if not is_evaluating:
            context_embed = torch.nn.functional.dropout(context_embed, p=0.75, training=False)

        flat_embed = context_embed.view(1, EMBEDDING_DIM * 3)
        flat_embed = torch.nn.functional.tanh(flat_embed.mm(transform_param))
        context_weight = flat_embed.mm(attention_param)  # (batch * max_contexts, 1)
        # 返回特征向量 dim = 383，和注意力 dim = 1
        return context_embed, context_weight


    def path_parser(self, path_array):
        context_embed_vec = []
        context_embed, context_weight_vec = self.line_parser(path_array[0])
        context_embed_vec.append(context_embed)
        for i in range(1, len(path_array)):
            try:
                context_embed, context_weight = self.line_parser(path_array[i])
            except:
                context_embed, context_weight = torch.FloatTensor(torch.zeros(EMBEDDING_DIM * 3, 1)), torch.FloatTensor(torch.zeros(1, 1))
            context_embed_vec.append(context_embed)
            context_weight_vec = torch.cat([context_weight_vec, context_weight], dim=1)
        attention_weights = torch.nn.functional.softmax(context_weight_vec)
        attention_weights = attention_weights.view(list(attention_weights.size())[1], 1)
        return context_embed_vec, attention_weights


    def get_tree_root(self):
        hash_tree = self._path_extractor.get_hash_tree()
        # 建立父节点和子节点的集合
        parent_set = set()
        children_set = set()
        for (parent, children) in hash_tree.items():
            if parent not in parent_set:
                parent_set.add(parent)
            for child in children:
                if child not in children_set:
                    children_set.add(child)
        diff = parent_set - children_set
        if len(diff) < 1:
            raise ValueError
        self._node_set = parent_set
        self._tree_root = diff.pop()

    def get_hash_tree(self):
        return self._path_extractor.get_hash_tree()


    @property
    def tree_root(self):
        return self._tree_root

    @property
    def node_vec_dict(self):
        return self._node_vec_dict








if __name__ == "__main__":
    path_analysis = PathAnalysis()
    #result, hash_to_string_dict = path_analysis._path_extractor.extract_paths('../Test.java')
    #print(path_analysis._path_extractor.get_paths_list())
    #print(path_analysis._path_extractor.get_hash_tree())
    #paths = path_analysis._path_extractor.get_paths_list()
    #path_analysis.path_parser(paths)
    #src = 'int,(PrimitiveType0)^(MethodDeclaration)_(NameExpr1),METHOD_NAME'
    path_analysis.predict_file("../Test.java")
    #context_embed, contexts_weights = path_analysis.line_parser(src)






