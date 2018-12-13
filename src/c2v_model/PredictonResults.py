class PredictionResults:
    def __init__(self, original_name):
        self.original_name = original_name
        self.predictions = list()
        self.attention_paths = list()

    def append_prediction(self, name, probability):
        self.predictions.append({'name': name, 'probability': probability})

    def append_attention_path(self, attention_score, token1, path, token2):
        self.attention_paths.append({'score': attention_score,
                                     'path': path,
                                     'token1': token1,
                                     'token2': token2})