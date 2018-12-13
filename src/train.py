import sys
import os
import random

import torch
import numpy as np


sys.path.append('/Users/shenxiaoang/Git/xLab/src/nn/')
sys.path.append('/Users/shenxiaoang/Git/xLab/src/c2v_model')

from NaiveAnalysis import  PathAnalysis
from model import TreeLSTM


# 定义全局变量
EMBED_DIM = 384
CLS_HIDDEN_LAYER_DIM_1 = 256
CLS_HIDDEN_LAYER_DIM_2 = 128
CLS_OUT_DIM = 1
TRAIN_EPOCH = 100
BATCH_SIZE = 10



class Trainer(object):
    def __init__(self, data_path, model_path):
        super(Trainer, self).__init__()
        self._data_path = data_path
        self._model_path = model_path
        self._file_set = None
        self._file_tag = None
        self._data_analysis = PathAnalysis(self._model_path)
        self._tree_lstm = TreeLSTM(EMBED_DIM, EMBED_DIM, CLS_HIDDEN_LAYER_DIM_1, CLS_HIDDEN_LAYER_DIM_2, CLS_OUT_DIM)

    def load_data(self, file_type):
        print('Loading data from %s' % (self._data_path))
        for root, dirs, files in os.walk(self._data_path):
            all_files = files
        self._file_set = []
        # 抽取出相应类型的文件
        for i in range(len(all_files)):
            if file_type in all_files[i]:
                self._file_set.append(all_files[i])
        # 加载文件标签
        self._file_tag = np.array([])
        # 统计正负样本的数量
        negSamp = 0
        posSamp = 0
        for i in range(0, len(self._file_set)):
            tag = int(self._file_set[i][-5])
            if tag == 0:
                posSamp += 1
            else:
                negSamp += 1
            self._file_tag = np.append(self._file_tag, tag)
        self._file_tag = torch.from_numpy(self._file_tag)
        self._file_tag = self._file_tag.view(list(self._file_tag.size())[0], 1)
        print('Loaded %d %s files from %s, including %d positive samples and %d negative samples.' % (len(self._file_set), file_type, self._data_path, posSamp, negSamp))


    def data_shuffle(self):

        self._file_tag = self._file_tag.view(list(self._file_tag.size())[0]).numpy().tolist()

        print(self._file_tag)

        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self._file_set)
        random.seed(randnum)
        random.shuffle(self._file_tag)
        self._file_tag = torch.from_numpy(np.array(self._file_tag))
        self._file_tag = self._file_tag.view(list(self._file_tag.size())[0], 1)



    def train(self):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self._tree_lstm.parameters(), lr=0.01)
        self.data_shuffle()
        for i in range(len(self._file_set)):
            print(self._file_set[i], i)
            self._data_analysis.predict_file(os.path.join(self._data_path, self._file_set[i]))
            _tree_dict = self._data_analysis.get_hash_tree()
            _tree_vec = self._data_analysis.node_vec_dict
            _tree_root = self._data_analysis.tree_root
            output = self._tree_lstm(_tree_dict, _tree_vec, _tree_root)
            print(output)
            optimizer.zero_grad()
            loss = criterion(output[0], self._file_tag[i].float())
            loss.backward()
            optimizer.step()
            print(loss)





# class Train(object):
#     def __init__(self, data_path, model_path):
#         self._data_path = data_path
#         self._model_path = model_path
#         self._data_analysis = PathAnalysis(self._model_path)
#         self._lstm_cell = TreeLSTMCell(TENSOR_DIM, HIDDEN_LAYER_DIM_1, HIDDEN_LAYER_DIM_2, tf.random_normal_initializer(mean=0, stddev=1))
#         self._file_set = None
#         self._file_tag = None
#         self._tree_root = None
#         self._hash_tree = None
#         self._node_vec_dict = None
#         self._train_index = 0
#         self._train_epoch = 1
#
#     def clear(self):
#         self._tree_root = None
#         self._hash_tree = None
#         self._node_vec_dict = None
#
#     def load_data(self, file_type):
#         print('Loading data from %s' % (self._data_path))
#         for root, dirs, files in os.walk(self._data_path):
#             all_files = files
#         self._file_set = np.array([])
#         # 抽取出相应类型的文件
#         for i in range(0, len(all_files)):
#             if file_type in all_files[i]:
#                 self._file_set = np.append(self._file_set, all_files[i])
#         # 加载文件标签
#         self._file_tag = np.array([])
#         # 统计正负样本的数量
#         negSamp = 0
#         posSamp = 0
#         for i in range(0, np.size(self._file_set)):
#             tag = int(self._file_set[i][0])
#             if tag == 0:
#                 posSamp += 1
#             else:
#                 negSamp += 1
#             self._file_tag = np.append(self._file_tag, tag)
#         print('Loaded %d %s files from %s, including %d positive samples and %d negative samples.' % (len(self._file_set), file_type, self._data_path, posSamp, negSamp))
#
#     def tree_cursor(self, node_index):
#         leaf_node = True
#         # 获取当前节点的特征向量，如果dict中不存在的话就直接返回零向量
#         try:
#             feature_tensor = self._node_vec_dict[node_index]
#         except:
#             return np.array([tf.Variable(tf.zeros((TENSOR_DIM, 1))), tf.Variable(tf.zeros((TENSOR_DIM, 1)))])
#         # 获取当前节点的子节点
#         children_node = self._hash_tree[node_index]
#         # 子节点输出的h\c值组成的np array
#         hc_input = None
#         for i in range(0, len(children_node)):
#             tmp_hc = np.array([self.tree_cursor(children_node[i])])
#             if i == 0:
#                 leaf_node = False
#                 hc_input = tmp_hc
#             else:
#                 hc_input = np.concatenate([hc_input, tmp_hc])
#         # 如果是叶节点的话直接返回零向量
#         if leaf_node:
#             hc_input = np.array([[tf.Variable(tf.zeros((TENSOR_DIM, 1))), tf.Variable(tf.zeros((TENSOR_DIM, 1)))]])
#         self._lstm_cell.init_inputs(hc_input, feature_tensor)
#         return self._lstm_cell.lstm_cell()
#
#     # 每次迭代都要调用一次该函数
#     def __call__(self):
#         if self._train_index >= len(self._file_set):
#             self._train_index = 0
#             self._train_epoch += 1
#         # 重置参数
#         self.clear()
#         # 数据预处理
#         self._data_analysis.predict_file(os.path.join(self._data_path, self._file_set[self._train_index]))
#         self._hash_tree = self._data_analysis.get_hash_tree()
#         self._node_vec_dict = self._data_analysis.node_vec_dict
#         self._tree_root = self._data_analysis.tree_root
#         # 构建训练图
#         tree_lstm_output = self.tree_cursor(self._tree_root)
#         out_put = self._lstm_cell.classifier(tree_lstm_output[0])
#         label = tf.constant(self._file_tag[self._train_index], dtype=tf.float32, shape=(1, 1))
#         return out_put, label
#
#     def get_train_batches(self):
#         return len(self._file_set)
#
if __name__ == "__main__":
    data_path = os.path.join('..', 'data', 'code_snippet', 'dataset')
    model_path = os.path.join('..', 'data', 'models', 'java14_model', 'saved_model_iter8.release')
    train = Trainer(data_path, model_path)
    train.load_data('.txt')
    train.train()



