import collections

import dataset

FilterSizes = collections.namedtuple(
    'FilterSizes', ['conv0', 'conv1', 'conv2', 'conv3'])


class Model:
    def __init__(
        self,
        name,
        datadir,
        validation_size=500,
        test_size=1,
        batch_size=100,
        learning_rate=0.01,
        learning_rate_decay_factor=.1,
        max_steps=1000,
        rnn_cell_size=64,
        num_rnn_layers=1,
        grad_clip=10,
        conv_filter_sizes=FilterSizes(0, 6, 6, 6),
        embedding_dims=dataset.EmbeddingSize(
            **{'chars': 5, 'fonts': 3, 'fontsizes': 2, 'tokens': 10}),
        use_lstm=False,
        use_rnn_layer_norm=False,
        dropout_keep_prob=1.0
    ):
        self.name = name
        self.data = dataset.read_datasets(datadir, validation_size, test_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.grad_clip = grad_clip,
        self.max_steps = max_steps
        self.rnn_cell_size = rnn_cell_size
        self.num_rnn_layers = num_rnn_layers
        self.feature_vocab_size = self.data.feature_vocab_size
        self.token_vocab_size = self.data.token_vocab_size
        self.filters = conv_filter_sizes
        self.embedding_dims = embedding_dims
        self.use_lstm = use_lstm
        self.use_rnn_layer_norm = use_rnn_layer_norm
        self.dropout_keep_prob = dropout_keep_prob
        self.feature_dim = (embedding_dims.chars +
                            embedding_dims.fonts +
                            embedding_dims.fontsizes)

    @classmethod
    def small(cls, datadir, validation_size=500, test_size=1):
        return cls('small', datadir, validation_size, test_size,
                   100, 1e-3, 1, 2000, 2, 1)

    @classmethod
    def medium(cls, datadir, validation_size=1000, test_size=1):
        return cls('medium', datadir, validation_size, test_size,
                   100, 0.0001, 1, 2000, 256, 3)

    @classmethod
    def large(cls, datadir, validation_size=5000, test_size=1):
        return cls('large', datadir, validation_size, test_size,
                   50, 0.0001, 1, 6000, 1536, 3,
                   conv_filter_sizes=FilterSizes(10, 10, 10, 10),
                   dropout_keep_prob=0.5)

    @classmethod
    def medium_reg(cls, datadir, validation_size=500, test_size=1):
        return cls('medium-reg', datadir, validation_size, test_size,
                   100, 0.01, 1, 3000, 200, 4,
                   use_lstm=True, use_rnn_layer_norm=True)
