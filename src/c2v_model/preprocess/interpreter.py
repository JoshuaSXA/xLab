import os
import re


class Interpreter:
    def __init__(self, file_path):
        self._data_path = file_path
        self._file_list = []
        for root, dirs, files in os.walk(self._data_path):
            # 存储了文件夹下所有的文件
            all_files = files
            print(all_files)

        # 抽取出.txt文件
        for i in range(0, len(all_files)):
            if '.txt' in all_files[i]:
                self._file_list.append(all_files[i])
        print(self._file_list)
        # 文件数量
        self._file_num = len(self._file_list)
        # 方法名
        self._method_name = ''
        # 用于存储将要返回的路径
        self._ret_vec = []
        # 用于存储hash code
        self._hash_code = []
        # dict,用于存储树的索引
        self._hash_tree = {}
        # 文件迭代索引，用于迭代
        self._file_index = 0

    def file_iterator(self):
        # 清空数据
        self._ret_vec = []
        self._hash_code = []
        self._hash_tree = {}
        # 处理
        self.data_handler()
        self._file_index += 1

    def data_handler(self):
        is_tree = False
        filename = os.path.join(self._data_path, self._file_list[self._file_index])
        f = open(filename)
        self._method_name = f.readline()
        line = f.readline()
        while line:
            if len(line.strip()) == 0:
                is_tree = True
                line = f.readline()
                continue
            if is_tree:
                self.tree_interpreter(line)
            else:
                self.path_interpreter(line)
            line = f.readline()
        # 这里需要对已经解析出来的数据进行存储
        f.close()

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

    @property
    def ret_vec(self):
        return self._ret_vec

    @property
    def hash_code(self):
        return self._hash_code

    @property
    def hash_tree(self):
        return self._hash_tree







