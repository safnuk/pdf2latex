import collections
import re

import tensorflow as tf

import dataset


FilterSizes = collections.namedtuple('FilterSizes', ['conv1', 'conv2', 'conv3'])


class Network:
    def __init__(
        self,
        batch_size,
        validation_size,
        test_size,
        learning_rate=0.1,
        learning_rate_decay_factor=.1,
        rnn_cell_size=10,
        num_rnn_layers=1,
        conv_filter_sizes=FilterSizes(5, 3, 5),
        embedding_dims=dataset.EmbeddingSize(
            **{'chars': 3, 'fonts': 2, 'fontsizes': 1, 'tokens': 10}),
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
        self.input_dim = (embedding_dims.chars + embedding_dims.fonts +
                          embedding_dims.fontsizes)
        self.use_lstm = use_lstm
        self.dtype = tf.float16 if use_fp16 else tf.float32
        self.data = dataset.read_datasets('data/', validation_size, test_size)
        self.tower_name = tower_name
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=self.dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

    def build(self):

        # Feeds for inputs.
        self.input_pdfs = tf.placeholder(tf.int32, shape=[None, 366, 100, 3],
                                         name='input')
        self.tokens = tf.placeholder(
            tf.int32, shape=[None, self.data.token_sequence_length],
            name='tokens'
        )
        e = self.embedding_layers(self.input_pdfs)
        c = self.convolution_layers(e)
        r = self.seq2seq_layer(c, self.tokens)
        self.rnn = self.loss_layer(r, self.tokens)

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
            kernel = self._variable_with_weight_decay(
                'weights', shape=[4, 6, self.input_dim, self.filters.conv1],
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

    def seq2seq_layer(self, conv, token_inputs):
        rnn_inputs = tf.reshape(conv, [self.batch_size, -1, self.filters.conv3])
        num_tokens = self.data.token_vocab_size
        with tf.variable_scope('encoder'):
            cell = self._rnn_cell()
            _, encoded_state = tf.nn.dynamic_rnn(cell, rnn_inputs,
                                                 dtype=self.dtype)
            self._activation_summary(encoded_state)

        with tf.variable_scope('embed_tokens') as scope:
            token_shape = tf.shape(token_inputs)
            slice_shape = [token_shape[0], 1]
            go_array = tf.fill(slice_shape, self.data.GO_TOKEN)
            tokens_with_GO = tf.concat([go_array, token_inputs], 1)
            embeddings = self._variable_on_cpu(
                'embeddings',
                [num_tokens, self.embedding_dims.tokens],
                tf.random_uniform_initializer(-1.0, 1.0)
            )
            token_embedding = tf.nn.embedding_lookup(
                embeddings, tokens_with_GO, name=scope.name)

        with tf.variable_scope('decoder'):
            cell = self._rnn_cell()
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_tokens)
            decoder_outputs, _ = tf.nn.dynamic_rnn(cell, token_embedding,
                                                   initial_state=encoded_state)
        return decoder_outputs

    def loss_layer(self, logits, targets):
        # http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
        with tf.name_scope("sequence_loss", [logits, targets]):
            slice_GO_token = tf.slice(
                logits, begin=[0, 1, 0], size=[-1, -1, -1])
            num_classes = tf.shape(slice_GO_token)[2]
            logits_flat = tf.reshape(slice_GO_token, [-1, num_classes])
            targets = tf.reshape(targets, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=logits_flat)
            total_loss = tf.reduce_mean(crossent)
            return total_loss
        # train_step = tf.train.AdagradOptimizer(
        #     learning_rate).minimize(total_loss)

    def _rnn_cell(self):
        if self.num_rnn_layers > 1:
            return tf.contrib.rnn.MultiRNNCell(
                [self._single_rnn_cell()
                 for _ in range(self.num_rnn_layers)])
        return self._single_rnn_cell()

    def _single_rnn_cell(self):
        if self.use_lstm:
            return tf.contrib.rnn.BasicLSTMCell(self.rnn_cell_size)
        return tf.contrib.rnn.GRUCell(self.rnn_cell_size)


def _dummy():
    pass


if __name__ == '__main__':
    network = Network(batch_size=100, validation_size=500, test_size=500)
    network.build()
    feature_batch, token_batch = network.data.train.next_batch(100)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # shape = tf.shape(network.rnn)
    print(network.rnn.eval(feed_dict={network.input_pdfs: feature_batch,
                                      network.tokens: token_batch}))
