import os
import re
import io
from preprocess.FeatureExtractor import Extractor


class SingleInterpreter(object):
    def path_interpreter(self, path):
        path = path.strip()
        # 提取字符串中的hash code
        hash_pattern = re.compile(r'(?<=[,\|])\d+(?=[\|\(])')
        it = hash_pattern.finditer(path)
        tmp_hash_unit = []
        for math_unit in it:
            tmp_hash_unit.append(math_unit.group())
        self._hash_code.append(tmp_hash_unit)
        repl_pattern = re.compile(r'(?<=,)[\d\|]+(?=\()')
        res_path = repl_pattern.sub('', path)
        self._ret_vec.append(res_path)


    def tree_interpreter(self, tree_seg):
        tree_seg = tree_seg.strip()
        split_list = tree_seg.split(', ')
        parent_node = split_list[0]
        child_node = split_list[1]
        if len(child_node.split(':')[1]):
            self._hash_tree[parent_node.split(': ')[1]] = child_node.split(': ')[1].strip('|').split('|')
        else:
            self._hash_tree[parent_node.split(': ')[1]] = []

    def data_handler(self, out):
        assert isinstance(out, str)
        is_tree = False
        out = io.StringIO(out)
        self._method_name = out.readline()
        line = out.readline()
        while line:
            if len(line.strip()) == 0:
                is_tree = True
                line = out.readline()
                continue
            if is_tree:
                self.tree_interpreter(line)
            else:
                self.path_interpreter(line)
            line = out.readline()
        # 这里需要对已经解析出来的数据进行存储
        out.close()

    def __init__(self):
        # 方法名
        self._method_name = ''
        # 用于存储将要返回的路径
        self._ret_vec = []
        # 用于存储hash code
        self._hash_code = []
        # dict,用于存储树的索引
        self._hash_tree = {}

    def __call__(self, out, err):
        if isinstance(out, bytes):
            out = out.decode()
        if isinstance(err, bytes):
            err = err.decode()
        assert isinstance(out, str)
        assert isinstance(err, str)
        self.__init__()
        if len(out) == 0:
            raise RuntimeError("Source file error:" + err)
        else:
            self.data_handler(out)
        result_con = str()
        result_con += self._method_name.replace("\n", " ")
        for s in self._ret_vec:
            result_con += s
            result_con += ' '
        return result_con

    @property
    def ret_vec(self):
        return self._ret_vec

    @property
    def hash_code(self):
        return self._hash_code

    @property
    def hash_tree(self):
        return self._hash_tree


if __name__ == "__main__":
    out, err = Extractor.extract("../../Test.java")
    interpreter = SingleInterpreter()
    print(interpreter(out.decode(), err.decode()))
