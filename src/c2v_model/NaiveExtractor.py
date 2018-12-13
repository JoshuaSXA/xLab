from preprocess.FeatureExtractor import Extractor
from preprocess.SingleInterpreter import SingleInterpreter
from Config import Config


class NaiveExtractor:
    def __init__(self, config):
        self.config = config
        self.interpreter = SingleInterpreter()

    def get_paths_list(self):
        return self.interpreter.ret_vec

    def get_hash_tree(self):
        return self.interpreter.hash_tree

    def get_path_hash(self):
        return self.interpreter.hash_code


    def extract_paths(self, path):
        out, err = Extractor.extract(path)
        self.interpreter(out, err)
        # Here we use the extracted paths from our extractor.
        # output = out.splitlines()
        # if len(output) == 0:
        #     err = err.decode()
        #     raise ValueError(err)
        # hash_to_string_dict = {}
        # result = []
        # for i, line in enumerate(output):
        #     parts = line.rstrip().split(' ')
        #     method_name = parts[0]
        #     current_result_line_parts = [method_name]
        #     contexts = parts[1:]
        #     for context in contexts[:self.config.MAX_CONTEXTS]:
        #         context_parts = context.split(',')
        #         context_word1 = context_parts[0]
        #         context_path = context_parts[1]
        #         context_word2 = context_parts[2]
        #         hashed_path = str(self.java_string_hashcode(context_path))
        #         hash_to_string_dict[hashed_path] = context_path
        #         current_result_line_parts += ['%s,%s,%s' % (context_word1, hashed_path, context_word2)]
        #     space_padding = ' ' * (self.config.MAX_CONTEXTS - len(contexts))
        #     result_line = ' '.join(current_result_line_parts) + space_padding
        #     result.append(result_line)
        # return result, hash_to_string_dict

    @staticmethod
    def java_string_hashcode(s):
        """
        Imitating Java's String#hashCode, because the model is trained on hashed paths but we wish to
        Present the path attention on un-hashed paths.
        """
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


if __name__ == "__main__":
    ne = NaiveExtractor(config=Config.get_default_config())
    ne.extract_paths("/Users/LeonWong/Desktop/Test.java")

