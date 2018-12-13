from interpreter import Interpreter
from FeatureExtractor import Extractor
import os


class DataProc(object):
    def __init__(self, data_path):
        # 源代码生成的txt文件的临时存放区域
        self._data_path = data_path
        self._extractor = Extractor()
        self._interpreter = Interpreter(self._data_path)
        self._path_vec = []

    # 生成txt文件
    def get_feature_path(self, file_path, file_name):
        out, _ = self._extractor.extract(os.path.join(file_path, file_name))
        out_path = os.path.join(self._data_path, file_name.replace('java', 'txt'))
        f = open(out_path, 'w')
        f.write(out.decode())
        f.close()

    # 从txt文件解析得到path
    def get_interpr_path(self):
        self._interpreter.file_iterator()
        self._path_vec = self._interpreter.ret_vec

    @property
    def path_vec(self):
        return self._path_vec


if __name__ == "__main__":
    obj = DataProc(os.path.join('..', '..', 'data', 'tree_path'))
    obj.get_feature_path('..', 'Test.java')
    obj.get_interpr_path()
