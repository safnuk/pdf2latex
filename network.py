import collections
import copy
import re

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib import legacy_seq2seq as seq2seq

import dataset


FilterSizes = collections.namedtuple('FilterSizes', ['conv1', 'conv2', 'conv3'])


class Network:
    def __init__(
        self,
        batch_size,
        validation_size,
        test_size,
        rnn_cell_size=10,
        num_rnn_layers=1,
        conv_filter_sizes=FilterSizes(5, 3, 5),
        embedding_dims=dataset.FeatureSize(
            **{'chars': 3, 'fonts': 2, 'fontsizes': 1}),
        use_lstm=False,
        data_dir='data/',
        use_fp16=False,
        tower_name='tower',
    ):
        self.batch_size = batch_size
        self.rnn_cell_size = rnn_cell_size
        self.num_rnn_layers = num_rnn_layers
        self.filters = conv_filter_sizes
        self.data_dir = data_dir
        self.embedding_dims = embedding_dims
        self.use_lstm = use_lstm
        self.dtype = tf.float16 if use_fp16 else tf.float32
        self.data = dataset.read_datasets('data/', validation_size, test_size)
        self.tower_name = tower_name

    def build(self):
        self.input_pdfs = tf.placeholder(tf.int32, shape=[None, 366, 100, 3])
        e = self.embedding_layers(self.input_pdfs)
        c = self.convolution_layers(e)
        self.rnn = self.seq2seq_layer(c)

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
            x: Tensor
        Returns:
            nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
        # training session. This helps the clarity on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % self.tower_name, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape,
                                  initializer=initializer, dtype=self.dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal
        distribution. A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float.
                 If None, weight decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=self.dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var),
                                       wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def embedding_layers(self, input_pdfs):
        """Embed the input data into dense vector representations.

        Args:
            pdfs: Input data returned from :func:`input`.
        Returns:
            Dense vector space representation of the input data.
        """
        slices = tf.unstack(input_pdfs, axis=3)
        vocab_size = self.data.feature_vocab_size
        with tf.variable_scope('char_embed') as scope:
            embeddings = self._variable_on_cpu(
                'embeddings',
                [vocab_size.chars, self.embedding_dims.chars],
                tf.random_uniform_initializer(-1.0, 1.0)
            )
            input_embed = tf.nn.embedding_lookup(embeddings,
                                                 slices[0], name=scope.name)
        with tf.variable_scope('font_embed') as scope:
            embeddings = self._variable_on_cpu(
                'embeddings',
                [vocab_size.fonts, self.embedding_dims.fonts],
                tf.random_uniform_initializer(-1.0, 1.0)
            )
            font_embed = tf.nn.embedding_lookup(embeddings,
                                                slices[1], name=scope.name)
        with tf.variable_scope('fontsize_embed') as scope:
            embeddings = self._variable_on_cpu(
                'embeddings',
                [vocab_size.fontsizes, self.embedding_dims.fontsizes],
                tf.random_uniform_initializer(-1.0, 1.0)
            )
            fontsize_embed = tf.nn.embedding_lookup(embeddings,
                                                    slices[2], name=scope.name)

        embedded = tf.concat([input_embed, font_embed, fontsize_embed], 3)
        return embedded

    def convolution_layers(self, embedded):
        """Build the convolution layers of the network.

        Args:
            embedded: Tensors returned from :func:`embedding_layers`.
        Returns:
            Application of three convolution layers.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPUs.
        # If we only ran this model on a single GPU, we could simplify this
        # function by replacing all instances of tf.get_variable()
        #  with tf.Variable().
        with tf.variable_scope('conv1') as scope:
            embedding_dim = sum(self.embedding_dims)
            kernel = self._variable_with_weight_decay(
                'weights', shape=[4, 6, embedding_dim, self.filters.conv1],
                stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(embedded, kernel, [1, 2, 2, 1], padding='VALID')
            biases = self._variable_on_cpu(
                'biases', [self.filters.conv1], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv1)
        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay(
                'weights', shape=[3, 6, self.filters.conv1, self.filters.conv2],
                stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv1, kernel, [1, 3, 3, 1], padding='VALID')
            biases = self._variable_on_cpu('biases', [self.filters.conv2],
                                           tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv2)
        with tf.variable_scope('conv3') as scope:
            kernel = self._variable_with_weight_decay(
                'weights', shape=[3, 3, self.filters.conv2, self.filters.conv3],
                stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(conv2, kernel,
                                [1, 1, 2, 1], padding='SAME')
            biases = self._variable_on_cpu(
                'biases', [self.filters.conv3], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv3)
        return conv3

    def seq2seq_layer(self, conv):
        return conv

    def _construct_seq2seq(self, cell, encoder_inputs,
                           decoder_inputs, feed_previous):
        with tf.variable_scope('seq2seq') as scope:
            encoder_cell = copy.deepcopy(cell)
            _, encoder_state = core_rnn.static_rnn(
                encoder_cell, encoder_inputs, dtype=self.dtype, scope=scope)

            cell = core_rnn_cell.OutputProjectionWrapper(cell, self.num_tokens)

            return seq2seq.embedding_rnn_decoder(
                decoder_inputs,
                encoder_state,
                cell,
                self.num_tokens,
                sum(self.embedding_dims),
                output_projection=None,
                feed_previous=feed_previous,
                scope=scope, dtype=self.dtype
            )

    def _rnn_cell(self):
        if self.num_rnn_layers > 1:
            return tf.contrib.rnn.MultiRNNCell(
                [self._single_rnn_cell() for _ in range(self.num_rnn_layers)])
        return self._single_rnn_cell

    def _single_rnn_cell(self):
        if self.use_lstm:
            return tf.contrib.rnn.BasicLSTMCell(self.rnn_cell_size)
        return tf.contrib.rnn.GRUCell(self.cell_size)


def _dummy():
    pass


if __name__ == '__main__':
    network = Network(batch_size=100, validation_size=500, test_size=500)
    network.build()
    batch, _ = network.data.train.next_batch(100)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    shape = tf.shape(network.rnn)
    print(shape.eval(feed_dict={network.input_pdfs: batch}))
