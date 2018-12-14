import sys
import os
import warnings
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
BATCH_SIZE = 20

warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self, train_data_path, test_data_path, model_path):
        super(Trainer, self).__init__()
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._model_path = model_path
        self._train_file_list = None
        self._test_file_list = None
        self._train_file_tag = None
        self._test_file_tag = None
        self._data_analysis = PathAnalysis(self._model_path)
        self._tree_lstm = TreeLSTM(EMBED_DIM, EMBED_DIM, CLS_HIDDEN_LAYER_DIM_1, CLS_HIDDEN_LAYER_DIM_2, CLS_OUT_DIM)

    def load_data(self, file_type, train=True):
        if train:
            data_path = self._train_data_path
        else:
            data_path = self._test_data_path
        print('Loading data from %s' % (data_path))
        for root, dirs, files in os.walk(data_path):
            all_files = files
        file_list = []
        # 抽取出相应类型的文件
        for i in range(len(all_files)):
            if file_type in all_files[i]:
                file_list.append(all_files[i])
        # 加载文件标签
        file_tag = np.array([])
        # 统计正负样本的数量
        negSamp = 0
        posSamp = 0
        for i in range(0, len(file_list)):
            tag = int(file_list[i][-5])
            if tag == 0:
                posSamp += 1
            else:
                negSamp += 1
            file_tag = np.append(file_tag, tag)
        file_tag = torch.from_numpy(file_tag)
        file_tag = file_tag.view(list(file_tag.size())[0], 1)
        print('Loaded %d %s files from %s, including %d positive samples and %d negative samples.' % (len(file_list), file_type, data_path, posSamp, negSamp))
        if train:
            self._train_file_list = file_list
            self._train_file_tag = file_tag
        else:
            self._test_file_list = file_list
            self._test_file_tag = file_tag


    def data_shuffle(self):
        self._train_file_tag = self._train_file_tag.view(list(self._train_file_tag.size())[0]).numpy().tolist()
        print(self._train_file_tag)
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self._train_file_list)
        random.seed(randnum)
        random.shuffle(self._train_file_tag)
        self._train_file_tag = torch.from_numpy(np.array(self._train_file_tag))
        self._train_file_tag = self._train_file_tag.view(list(self._train_file_tag.size())[0], 1)


    def train(self):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self._tree_lstm.parameters(), lr=0.01)
        optimizer.zero_grad()
        total_loss = 0.0
        self.data_shuffle()
        for i in range(len(self._train_file_list)):
            self._data_analysis.predict_file(os.path.join(self._train_data_path, self._train_file_list[i]))
            _tree_dict = self._data_analysis.get_hash_tree()
            _tree_vec = self._data_analysis.node_vec_dict
            _tree_root = self._data_analysis.tree_root
            output = self._tree_lstm(_tree_dict, _tree_vec, _tree_root)
            loss = criterion(output[0], self._train_file_tag[i].float())
            total_loss += loss.item()
            loss.backward()
            if (i + 1) % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(total_loss)
        return total_loss / len(self._train_file_list)

    def test(self):
        self._tree_lstm.eval()
        criterion = torch.nn.BCELoss()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.FloatTensor(torch.zeros(len(self._test_file_list), 1))
            for i in range(len(self._test_file_list)):
                self._data_analysis.predict_file(os.path.join(self._test_data_path, self._test_file_list[i]))
                _tree_dict = self._data_analysis.get_hash_tree()
                _tree_vec = self._data_analysis.node_vec_dict
                _tree_root = self._data_analysis.tree_root
                output = self._tree_lstm(_tree_dict, _tree_vec, _tree_root)
                #print(output, self._test_file_tag[i])
                predictions[i] = output.ge(0.5).float()
                loss = criterion(output[0], self._test_file_tag[i].float())
                total_loss += loss.item()
            accuracy = 1 - (predictions - self._test_file_tag.float()).abs_().sum() / len(self._test_file_tag)
        return total_loss / len(self._test_file_list), accuracy


if __name__ == "__main__":
    train_data_path = os.path.join('..', 'data', 'code_snippet', 'Dataset2018', 'train_data')
    test_data_path = os.path.join('..', 'data', 'code_snippet', 'Dataset2018', 'test_data')
    model_path = os.path.join('..', 'data', 'models', 'java14_model', 'saved_model_iter8.release')
    train = Trainer(train_data_path, test_data_path, model_path)
    train.load_data('.txt')
    train.load_data('.txt', False)
    for i in range(TRAIN_EPOCH):
        train_loss = train.train()
        test_loss, accuracy = train.test()
        print("EPOCH %d: train loss is %f, test loss is %f, test accuracy is %f" % ((i + 1), train_loss, test_loss, accuracy))
        torch.save(train._tree_lstm.state_dict(), os.path.join('..', 'data', 'models', 'tree_lstm_model', 'model_epoch_%d.pkl' % (i + 1)))



