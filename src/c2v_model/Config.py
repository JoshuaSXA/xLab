class Config(object):
    @staticmethod
    def get_default_config(data_path=None, test_path=None, save_path=None, load_path="../../data/models/java14_model/saved_model_iter8.release", release=True):
        config = Config()
        config.NUM_EPOCHS = 20
        config.SAVE_EVERY_EPOCHS = 1
        config.BATCH_SIZE = 1024
        config.TEST_BATCH_SIZE = config.BATCH_SIZE
        config.READING_BATCH_SIZE = 1300 * 4
        config.NUM_BATCHING_THREADS = 2
        config.BATCH_QUEUE_SIZE = 300000
        config.MAX_CONTEXTS = 200
        config.WORDS_VOCAB_SIZE = 1301136
        config.TARGET_VOCAB_SIZE = 261245
        config.PATHS_VOCAB_SIZE = 911417
        config.EMBEDDINGS_SIZE = 128
        config.MAX_TO_KEEP = 10
        # Automatically filled, do not edit:
        config.TRAIN_PATH = data_path
        config.TEST_PATH = test_path
        config.SAVE_PATH = save_path
        config.LOAD_PATH = load_path
        config.RELEASE = release
        return config

    def __init__(self):
        self.NUM_EPOCHS = 0
        self.SAVE_EVERY_EPOCHS = 0
        self.BATCH_SIZE = 0
        self.TEST_BATCH_SIZE = 0
        self.READING_BATCH_SIZE = 0
        self.NUM_BATCHING_THREADS = 0
        self.BATCH_QUEUE_SIZE = 0
        self.TRAIN_PATH = ''
        self.TEST_PATH = ''
        self.MAX_CONTEXTS = 0
        self.WORDS_VOCAB_SIZE = 0
        self.TARGET_VOCAB_SIZE = 0
        self.PATHS_VOCAB_SIZE = 0
        self.EMBEDDINGS_SIZE = 0
        self.SAVE_PATH = ''
        self.LOAD_PATH = ''
        self.MAX_TO_KEEP = 0
        self.RELEASE = False