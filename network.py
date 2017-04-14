import collections
import os
import shutil

import tensorflow as tf

import dataset
import scope

FilterSizes = collections.namedtuple('FilterSizes', ['conv1', 'conv2', 'conv3'])


class Network:
    def __init__(self, model):
        self.model = model
        self.dropout = 1.0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.input = dataset.Batch(
            tf.placeholder(tf.int32, shape=[None, 366, 100, 3]),
            tf.placeholder(tf.int32,
                           shape=[None, self.model.data.token_sequence_length])
        )
        self.feature_embedding
        self.convolution
        self.encoder
        self.token_embedding
        self.decoder
        self.logits
        self.loss
        self.accuracy
        self.optimize

    @scope.lazy_load
    def feature_embedding(self):
        """Embed the input data into dense vector representations.

        Returns:
            Dense vector space representation of the input data.
        """
        vocab_size = self.model.feature_vocab_size
        slices = tf.unstack(self.input.pdf, axis=3, name='slice')
        with tf.variable_scope('char'):
            char = self._embedding_layer(
                slices[0], [vocab_size.chars, self.model.embedding_dims.chars])
        with tf.variable_scope('font'):
            font = self._embedding_layer(
                slices[1], [vocab_size.fonts, self.model.embedding_dims.fonts])
        with tf.variable_scope('fontsize'):
            fontsize = self._embedding_layer(
                slices[2],
                [vocab_size.fontsizes, self.model.embedding_dims.fontsizes]
            )
        embedded = tf.concat([char, font, fontsize], 3)
        return embedded

    @scope.lazy_load
    def convolution(self):
        """Build the convolution layers of the network.

        Returns:
            Application of three convolution layers to
            the feature_embedding layer.
        """
        with tf.variable_scope('filter1'):
            conv1 = self._conv_relu(
                self.feature_embedding,
                [4, 6, self.model.feature_dim, self.model.filters.conv1],
                [1, 2, 2, 1], 'VALID')
        with tf.variable_scope('filter2'):
            conv2 = self._conv_relu(
                conv1,
                [3, 6, self.model.filters.conv1, self.model.filters.conv2],
                [1, 3, 3, 1], 'VALID')
        with tf.variable_scope('filter3'):
            conv3 = self._conv_relu(
                conv2,
                [3, 3, self.model.filters.conv2, self.model.filters.conv3],
                [1, 1, 2, 1], 'SAME')
        return conv3

    @scope.lazy_load
    def encoder(self):
        conv = self.convolution
        conv_shape = tf.shape(conv)
        # flatten to a sequence of vectors to feed to the encoder
        rnn_inputs = tf.reshape(
            conv, [conv_shape[0], -1, self.model.filters.conv3], name='flatten')
        # reverse to feed the sequence in reverse order - last chars to first
        rnn_reversed = tf.reverse(rnn_inputs, [-1], name='reverse')
        cell = self._rnn_cell()
        _, encoded_state = tf.nn.dynamic_rnn(
            cell, rnn_reversed, dtype=tf.float32)
        return encoded_state

    @scope.lazy_load
    def token_embedding(self):
        num_tokens = self.model.token_vocab_size
        token_shape = tf.shape(self.input.token)
        self.target_sequences, lengths = tf.split(
            self.input.token, [token_shape[1] - 1, 1], axis=1)
        self.sequence_lengths = tf.squeeze(lengths)
        token_embedding = self._embedding_layer(
            self.target_sequences,
            [num_tokens, self.model.embedding_dims.tokens]
        )
        return token_embedding

    @scope.lazy_load
    def decoder(self):
        cell = self._rnn_cell()
        time_major = tf.transpose(self.token_embedding, [1, 0, 2])
        W = tf.get_variable(
            'W', [self.model.rnn_cell_size, self.model.embedding_dims.tokens],
            initializer=tf.truncated_normal_initializer()
        )
        b = tf.get_variable('b', [self.model.embedding_dims.tokens],
                            initializer=tf.constant_initializer(0.0))

        def fn(a, x):
            state1, state2 = cell(x, a[1])
            return (tf.reshape(
                tf.matmul(state1, W) + b,
                [self.model.batch_size, self.model.embedding_dims.tokens]),
                state2)

        initial_input = tf.zeros(
            [self.model.batch_size, self.model.embedding_dims.tokens],
            dtype=tf.float32)
        decoder_output, final_state = tf.scan(
            fn, time_major, initializer=(initial_input, self.encoder))
        return tf.transpose(decoder_output, [1, 0, 2])

    @scope.lazy_load
    def logits(self):
        W = tf.get_variable(
            'W', [self.model.embedding_dims.tokens,
                  self.model.token_vocab_size],
            initializer=tf.truncated_normal_initializer()
        )
        b = tf.get_variable('b', [self.model.token_vocab_size],
                            initializer=tf.constant_initializer(0.0))
        decoded_flat = tf.reshape(self.decoder,
                                  [-1, self.model.embedding_dims.tokens])
        return tf.matmul(decoded_flat, W) + b

    @scope.lazy_load
    def loss(self):
        targets_flat = tf.reshape(self.target_sequences, [-1], name='flatten')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_flat, logits=self.logits)
        total_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', total_loss)
        return total_loss

    @scope.lazy_load('loss')
    def accuracy(self):
        targets_flat = tf.reshape(self.target_sequences, [-1], name='flatten')
        # targets_flat = tf.get_default_graph() \
        #     .get_operation_by_name('loss/flatten')
        correct_prediction = tf.equal(
            tf.cast(tf.argmax(self.logits, 1), tf.int32),
            targets_flat, name='correct')
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    @scope.lazy_load
    def optimize(self):
        train_step = tf.train.AdamOptimizer(self.model.learning_rate)\
            .minimize(self.loss, global_step=self.global_step)
        return train_step

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
            x: Tensor
        Returns:
            nothing
        """
        with tf.name_scope('summaries'):
            tensor_name = x.op.name
            tf.summary.histogram(tensor_name + '/activations', x)
            tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _conv_relu(self, input, kernel_shape, strides, padding):
        weights = tf.get_variable(
            "weights", kernel_shape, initializer=tf.random_normal_initializer())
        self._activation_summary(weights)
        biases = tf.get_variable(
            "biases", kernel_shape[3], initializer=tf.constant_initializer(0.1))
        self._activation_summary(biases)
        conv = tf.nn.conv2d(
            input, weights, strides=strides, padding=padding)
        act = tf.nn.relu(conv + biases)
        self._activation_summary(act)
        return act

    def _embedding_layer(self, input, shape):
        """
        Constructs an embedding operation to embed a tensor of discrete features
        into a dense vector representation.

        Args:
            input: vector of features to be embedded
            shape: 2D-tensor with values [vocab_size, embedding_dimensions]
        Returns:
            Graph operation that transforms input into a tensor of
            embedded vectors.
        """
        with tf.variable_scope('embed'):
            matrix = tf.get_variable(
                'matrix', shape, tf.float32,
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self._activation_summary(matrix)
            return tf.nn.embedding_lookup(matrix, input, name='project')

    def _rnn_cell(self):
        if self.model.num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                [self._single_rnn_cell()
                 for _ in range(self.model.num_rnn_layers)])
        else:
            cell = self._single_rnn_cell()
        return tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=self.dropout)

    def _single_rnn_cell(self):
        if self.model.use_lstm:
            if self.model.use_rnn_layer_norm:
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    self.model.rnn_cell_size, layer_norm=True,
                    dropout_keep_prob=1.0)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.model.rnn_cell_size)
        else:
            cell = tf.contrib.rnn.GRUCell(self.model.rnn_cell_size)
        return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout)


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
        conv_filter_sizes=FilterSizes(5, 3, 5),
        embedding_dims=dataset.EmbeddingSize(
            **{'chars': 5, 'fonts': 3, 'fontsizes': 1, 'tokens': 10}),
        use_lstm=False,
        use_rnn_layer_norm=False,
        dropout_keep_prob=1.0
    ):
        self.name = name
        self.data = dataset.read_datasets(datadir, validation_size, test_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
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

    def feed_dict(self, input_feed, validate=False):
        batch = self.data.train.next_batch(self.batch_size)
        return {input_feed.pdf: batch.pdf, input_feed.token: batch.token}

    @classmethod
    def small(cls, datadir, validation_size=500, test_size=1):
        return cls('small', datadir, validation_size, test_size,
                   100, 0.01, 1, 400, 64, 1)

    @classmethod
    def medium(cls, datadir, validation_size=500, test_size=1):
        return cls('medium', validation_size, test_size,
                   100, 0.001, 1, 2000, 128, 4)

    @classmethod
    def medium_reg(cls, datadir, validation_size=500, test_size=1):
        return cls('medium-reg', validation_size, test_size,
                   100, 0.01, 1, 3000, 200, 4,
                   use_lstm=True, use_rnn_layer_norm=True)


def train(logdir, datadir, clear_old_logs=True):
    logdir = logdir + 'small/'
    if clear_old_logs:
        if os.path.exists(logdir):
            shutil.rmtree(logdir)

    with tf.Graph().as_default() as graph:
        network = Network(Model.small(datadir))
        summaries = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=logdir, summary_op=None, graph=graph)
    with sv.managed_session() as sess:

        # run until training should stop
        rounds = 0
        while not sv.should_stop():
            loss, acc, s,  _ = sess.run(
                [network.loss, network.accuracy, summaries,
                 network.optimize],
                feed_dict=network.model.feed_dict(network.input))

            # hand over your own summaries to the Supervisor
            sv.summary_computed(sess, s)

            if rounds % 10 == 0:
                print(
                    'Round {} Loss: {} Accuracy: {}'.format(rounds, loss, acc))

            if rounds > network.model.max_steps:
                sv.request_stop()
            rounds += 1

if __name__ == '__main__':
    # BASE = '/Users/safnu1b/Documents/latex/'
    BASE1 = ''
    BASE = '/data/safnu1b/latex/'
    datadir = BASE1 + 'data/'
    logdir = BASE + 'log/'
    train(logdir, datadir)
