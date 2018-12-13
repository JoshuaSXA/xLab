import re
from enum import Enum
from PredictonResults import PredictionResults

class Common(object):
    noSuchWord = "NoSuchWord"

    @staticmethod
    def normalize_word(word):
        stripped = re.sub(r'[^a-zA-Z]', '', word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def _load_vocab_from_histogram(path, min_count=0, start_from=0, return_counts=False):
        with open(path, 'r') as file:
            word_to_index = {}
            index_to_word = {}
            word_to_count = {}
            next_index = start_from
            for line in file:
                line_values = line.rstrip().split(' ')
                if len(line_values) != 2:
                    continue
                word = line_values[0]
                count = int(line_values[1])
                if count < min_count:
                    continue
                if word in word_to_index:
                    continue
                word_to_index[word] = next_index
                index_to_word[next_index] = word
                word_to_count[word] = count
                next_index += 1
        result = word_to_index, index_to_word, next_index - start_from
        if return_counts:
            result = (*result, word_to_count)
        return result

    @staticmethod
    def _load_vocab_from_dict(word_to_count, min_count=0, start_from=0):
        word_to_index = {}
        index_to_word = {}
        next_index = start_from
        for word, count in word_to_count.items():
            if count < min_count:
                continue
            if word in word_to_index:
                continue
            word_to_index[word] = next_index
            index_to_word[next_index] = word
            word_to_count[word] = count
            next_index += 1
        return word_to_index, index_to_word, next_index - start_from

    @staticmethod
    def load_vocab_from_histogram(path, min_count=0, start_from=0, max_size=None, return_counts=False):
        if max_size is not None:
            word_to_index, index_to_word, next_index, word_to_count = \
                Common._load_vocab_from_histogram(path, min_count, start_from, return_counts=True)
            if next_index <= max_size:
                results = (word_to_index, index_to_word, next_index)
                if return_counts:
                    results = (*results, word_to_count)
                return results
            # Take min_count to be one plus the count of the max_size'th word
            min_count = sorted(word_to_count.values(), reverse=True)[max_size] + 1
        return Common._load_vocab_from_histogram(path, min_count, start_from, return_counts)

    @staticmethod
    def load_vocab_from_dict(word_to_count, max_size=None, start_from=0):
        if max_size is not None:
            if max_size > len(word_to_count):
                min_count = 0
            else:
                min_count = sorted(word_to_count.values(), reverse=True)[max_size] + 1
        return Common._load_vocab_from_dict(word_to_count, min_count, start_from)

    @staticmethod
    def load_json(json_file):
        data = []
        with open(json_file, 'r') as file:
            for line in file:
                current_program = Common.process_single_json_line(line)
                if current_program is None:
                    continue
                for element, scope in current_program.items():
                    data.append((element, scope))
        return data

    @staticmethod
    def load_json_streaming(json_file):
        with open(json_file, 'r') as file:
            for line in file:
                current_program = Common.process_single_json_line(line)
                if current_program is None:
                    continue
                for element, scope in current_program.items():
                    yield (element, scope)

    @staticmethod
    def save_word2vec_file(file, vocab_size, dimension, index_to_word, vectors):
        file.write('%d %d\n' % (vocab_size, dimension))
        for i in range(1, vocab_size + 1):
            if i in index_to_word:
                file.write(index_to_word[i] + ' ')
                file.write(' '.join(map(str, vectors[i])) + '\n')

    @staticmethod
    def calculate_max_contexts(file):
        contexts_per_word = Common.process_test_input(file)
        return max(
            [max(l, default=0) for l in [[len(contexts) for contexts in prog.values()] for prog in contexts_per_word]],
            default=0)

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [Common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [Common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def load_file_lines(path):
        with open(path, 'r') as f:
            return f.read().splitlines()

    @staticmethod
    def split_to_batches(data_lines, batch_size):
        print('here is split_batches')
        return  [data_lines[x:x + batch_size] for x in range(0, len(data_lines), batch_size)]

    @staticmethod
    def legal_method_names_checker(name):
        return name != Common.noSuchWord and re.match('^[a-zA-Z\|]+$', name)

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(Common.legal_method_names_checker, top_words))
        return result

    @staticmethod
    def get_subtokens(str):
        return str.split('|')

    @staticmethod
    def parse_results(result, unhash_dict, topk=5):
        prediction_results = []
        for single_method in result:
            original_name, top_suggestions, top_scores, attention_per_context = list(single_method)
            current_method_prediction_results = PredictionResults(original_name)
            for i, predicted in enumerate(top_suggestions):
                if predicted == Common.noSuchWord:
                    continue
                suggestion_subtokens = Common.get_subtokens(predicted)
                current_method_prediction_results.append_prediction(suggestion_subtokens, top_scores[i].item())
            for context, attention in [(key, attention_per_context[key]) for key in
                                       sorted(attention_per_context, key=attention_per_context.get, reverse=True)][
                                      :topk]:
                token1, hashed_path, token2 = context
                if hashed_path in unhash_dict:
                    unhashed_path = unhash_dict[hashed_path]
                    current_method_prediction_results.append_attention_path(attention.item(), token1=token1,
                                                                            path=unhashed_path, token2=token2)
            prediction_results.append(current_method_prediction_results)
        return prediction_results