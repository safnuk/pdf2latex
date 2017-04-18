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
        self.targets
        self.convolution
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
    def targets(self):
        token_shape = tf.shape(self.input.token)
        # we don't need <GO> token for this network model
        _, targets, lengths = tf.split(
            self.input.token, [1, token_shape[1] - 2, 1], axis=1)
        # self.sequence_lengths = tf.squeeze(lengths)
        self.sequence_length = token_shape[1] - 2
        time_major = tf.transpose(targets, [1, 0])
        return tf.reshape(time_major, [-1], name='flatten')

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
        conv_shape = tf.shape(conv3)
        # flatten to a sequence of vectors to feed to the encoder
        rnn_inputs = tf.reshape(
            conv3, [conv_shape[0], -1, self.model.filters.conv3],
            name='flatten')
        inputs_time_major = tf.transpose(rnn_inputs, [1, 0, 2])
        pad_amount = self.sequence_length - tf.shape(inputs_time_major)[0]
        padded_to_length = tf.pad(inputs_time_major,
                                  [[0, pad_amount], [0, 0], [0, 0]],
                                  name='pad')
        # doesn't actually reshape, just provides shape info to future steps
        padded_to_length = tf.reshape(
            padded_to_length,
            [-1, self.model.batch_size, self.model.filters.conv3])
        return padded_to_length

    @scope.lazy_load
    def decoder(self):
        cell = self._rnn_cell()
        initial_state = cell.zero_state(self.model.batch_size, dtype=tf.float32)
        output = tf.scan(
            lambda a, x: cell(x, a)[1],
            self.convolution,
            initializer=initial_state)
        return output

    @scope.lazy_load
    def logits(self):
        # reshape/flatten into shape
        # [sequence_length x batch_size, rnn_cell_size * layers ]
        if self.model.num_rnn_layers == 1:
            decoded = (self.decoder)
        else:
            decoded = tf.transpose(self.decoder, [1, 2, 0, 3])
        decoded_dim = self.model.num_rnn_layers * self.model.rnn_cell_size
        decoded = tf.reshape(
            decoded,
            [-1, decoded_dim])
        W = tf.get_variable(
            'weight',
            [decoded_dim, self.model.token_vocab_size],
            initializer=tf.truncated_normal_initializer()
        )
        self._activation_summary(W)
        b = tf.get_variable('bias', [self.model.token_vocab_size],
                            initializer=tf.constant_initializer(0.0))
        self._activation_summary(b)
        return tf.matmul(decoded, W) + b

    @scope.lazy_load_no_scope
    def loss(self):
        with tf.variable_scope('loss') as scope:
            self.loss_scope = scope
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets, logits=self.logits)
            total_loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross_entropy', total_loss,
                              collections=['train', 'test'])
            return total_loss

    @scope.lazy_load_no_scope
    def accuracy(self):
        with tf.variable_scope(self.loss_scope.original_name_scope):
            correct_prediction = tf.equal(
                tf.cast(tf.argmax(self.logits, 1), tf.int32),
                self.targets, name='correct')
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy,
                              collections=['train', 'test'])
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
        # with tf.name_scope('summaries'):
        tensor_name = x.op.name
        tf.summary.histogram(
            tensor_name + '/activations', x, collections=['train'])
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x),
                          collections=['train'])

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

    @classmethod
    def small(cls, datadir, validation_size=500, test_size=1):
        return cls('small', datadir, validation_size, test_size,
                   100, 1e-3, 1, 2000, 64, 1)

    @classmethod
    def medium(cls, datadir, validation_size=100, test_size=1):
        return cls('medium', datadir, validation_size, test_size,
                   100, 0.0001, 1, 2000, 256, 4)

    @classmethod
    def medium_reg(cls, datadir, validation_size=500, test_size=1):
        return cls('medium-reg', datadir, validation_size, test_size,
                   100, 0.01, 1, 3000, 200, 4,
                   use_lstm=True, use_rnn_layer_norm=True)


def train(model, logdir, clear_old_logs=True):
    logdir = logdir + model.name + '/'
    if clear_old_logs:
        if os.path.exists(logdir):
            shutil.rmtree(logdir)

    def feed_dict(train=True):
        batch = network.model.data.train.next_batch(network.model.batch_size)
        return {
            network.input.pdf: batch.pdf,
            network.input.token: batch.token,
        }

    network = Network(model)
    train_summaries = tf.summary.merge_all('train')
    infer_summaries = tf.summary.merge_all('test')

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(logdir + 'train/', sess.graph)
    infer_writer = tf.summary.FileWriter(logdir + 'infer/')
    tf.global_variables_initializer().run()

    for n in range(network.model.max_steps):
        if n % 10 == 0:
            loss, acc, s, step = sess.run(
                [network.loss, network.accuracy, infer_summaries,
                 network.global_step],
                feed_dict=feed_dict(train=False))
            print(
                'Round {} Loss: {} Inference accuracy: {}'
                .format(n, loss, acc))
            infer_writer.add_summary(s, global_step=step)
        elif n % 100 == 99:  # Save model and record execution stats
            saver.save(sess, logdir + 'checkpoint-',
                       global_step=step)
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            loss, acc, s,  step, _ = sess.run(
                [network.loss, network.accuracy, train_summaries,
                    network.global_step, network.optimize],
                feed_dict=feed_dict(train=True),
                options=run_options,
                run_metadata=run_metadata)
            print('Adding run metadata for', n)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % n)
            train_writer.add_summary(s, global_step=step)
        else:
            loss, acc, s,  step, _ = sess.run(
                [network.loss, network.accuracy, train_summaries,
                    network.global_step, network.optimize],
                feed_dict=feed_dict(train=True))
            train_writer.add_summary(s, global_step=step)

    train_writer.close()
    infer_writer.close()

if __name__ == '__main__':
    # BASE = '/data/safnu1b/latex/'
    BASE = '/Users/safnu1b/Documents/latex/'
    BASE1 = ''
    datadir = BASE1 + 'data/'
    logdir = BASE + 'log/'
    model = Model.small(datadir)
    train(model, logdir, False)
