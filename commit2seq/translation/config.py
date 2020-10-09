class Config:
    @staticmethod
    def get_default_config(args):
        config = Config(args)
        config.NUM_EPOCHS = 3000
        config.SAVE_EVERY_EPOCHS = 1
        config.PATIENCE = 20
        config.BATCH_SIZE = 256
        config.TEST_BATCH_SIZE = 64
        config.READER_NUM_PARALLEL_BATCHES = 1
        config.SHUFFLE_BUFFER_SIZE = 10000
        config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
        config.MAX_CONTEXTS = int(args.path_num)
        config.EMBEDDINGS_SIZE = int(args.embed)
        config.RNN_SIZE = int(args.encoder)
        config.DECODER_SIZE = int(args.decoder)
        config.NUM_DECODER_LAYERS = 2
        config.MAX_PATH_LENGTH = 10
        config.MAX_NAME_PARTS = 8
        config.MAX_TARGET_PARTS = 10
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.4
        config.RNN_DROPOUT_KEEP_PROB = 0.4
        config.BIRNN = True
        config.RANDOM_CONTEXTS = True
        config.BEAM_WIDTH = 5
        config.USE_MOMENTUM = False
        config.INITIAL_LR = 0.0001
        return config

    def take_model_hyperparams_from(self, otherConfig):
        self.EMBEDDINGS_SIZE = otherConfig.EMBEDDINGS_SIZE
        self.RNN_SIZE = otherConfig.RNN_SIZE
        self.DECODER_SIZE = otherConfig.DECODER_SIZE
        self.NUM_DECODER_LAYERS = otherConfig.NUM_DECODER_LAYERS
        self.BIRNN = otherConfig.BIRNN

    def __init__(self, args):
        self.NUM_EPOCHS = 0
        self.SAVE_EVERY_EPOCHS = 0
        self.PATIENCE = 0
        self.BATCH_SIZE = 0
        self.TEST_BATCH_SIZE = 0
        self.READER_NUM_PARALLEL_BATCHES = 0
        self.SHUFFLE_BUFFER_SIZE = 0
        self.CSV_BUFFER_SIZE = None
        self.TRAIN_PATH = args.data_path
        self.TEST_PATH = args.test_path if args.test_path is not None else ''
        self.MAX_CONTEXTS = 0
        self.SUBTOKENS_VOCAB_MAX_SIZE = 0
        self.TARGET_VOCAB_MAX_SIZE = 0
        self.EMBEDDINGS_SIZE = 0
        self.RNN_SIZE = 0
        self.DECODER_SIZE = 0
        self.NUM_DECODER_LAYERS = 0
        self.SAVE_PATH = args.save_path_prefix
        self.LOAD_PATH = args.load_path
        self.MAX_PATH_LENGTH = 0
        self.MAX_NAME_PARTS = 0
        self.MAX_TARGET_PARTS = 0
        self.MAX_DISMATCHED_PARTS = 0
        self.EMBEDDINGS_DROPOUT_KEEP_PROB = 0
        self.RNN_DROPOUT_KEEP_PROB = 0
        self.BIRNN = False
        self.RANDOM_CONTEXTS = True
        self.BEAM_WIDTH = 1
        self.USE_MOMENTUM = True