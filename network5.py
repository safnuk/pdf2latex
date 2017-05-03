import tensorflow as tf

import dataset
import scope


class Network:
    def __init__(self, model):
        self.model = model
        self.dropout = tf.placeholder(tf.float32, shape=[])
        self.sparsity_adjustment = 0.9
        self.total_loss_ratio = 0.5  # portion of loss coming from autoencoder
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.input = dataset.Batch(
            tf.placeholder(tf.int32, shape=[None, 366, 100, 3]),
            tf.placeholder(tf.int32,
                           shape=[None, self.model.data.token_sequence_length])
        )
        self.feature_embedding
        self.targets
        self.convolution
        # self.autoencoder
        # self.autoencoder_loss
        self.encoded
        self.decoder
        self.logits
        self.loss
        # self.total_loss
        self.accuracy
        self.optimize

    @scope.lazy_load
    def trim(self):
        shape = tf.shape(self.input.pdf)
        trimmed = tf.slice(
            self.input.pdf, [0, 0, 0, 0], [-1, shape[1]-6, shape[2]-4, -1])
        trimmed.set_shape([self.model.batch_size, 360, 96, 3])
        return trimmed

    @scope.lazy_load
    def feature_embedding(self):
        """Embed the input data into dense vector representations.

        Returns:
            Dense vector space representation of the input data.
        """
        vocab_size = self.model.feature_vocab_size
        slices = tf.unstack(self.trim, axis=3, name='slice')
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
        self.sequence_lengths = tf.squeeze(lengths)
        self.sequence_length = token_shape[1] - 2
        return tf.reshape(targets, [-1], name='flatten')

    @scope.lazy_load
    def convolution(self):
        """Build the convolution layers of the network.

        Returns:
            Application of three convolution layers to
            the feature_embedding layer.
        """
        with tf.variable_scope('filter0'):
            if self.model.filters.conv0 > 0:
                conv0_dim = self.model.filters.conv0
                conv0 = self._conv_relu(
                    self.feature_embedding,
                    [5, 5, self.model.feature_dim, self.model.filters.conv0],
                    [1, 1, 1, 1], 'SAME')
            else:
                conv0 = self.feature_embedding
                conv0_dim = self.model.feature_dim
        with tf.variable_scope('filter1'):
            conv1 = self._conv_relu(
                conv0,
                [5, 5, conv0_dim, self.model.filters.conv1],
                [1, 1, 1, 1], 'SAME')
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('filter2'):
            conv2 = self._conv_relu(
                pool1,
                [5, 5, self.model.filters.conv1, self.model.filters.conv2],
                [1, 1, 1, 1], 'SAME')
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                                   strides=[1, 3, 3, 1], padding='SAME')
        with tf.variable_scope('filter3'):
            conv3 = self._conv_relu(
                pool2,
                [5, 5, self.model.filters.conv2, self.model.filters.conv3],
                [1, 1, 2, 1], 'SAME')
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 1, 2, 1],
                                   strides=[1, 1, 2, 1], padding='SAME')
        return pool3

    @scope.lazy_load
    def autoencoder(self):
        vocab_sizes = self.model.feature_vocab_size
        out_dim = vocab_sizes.chars + vocab_sizes.fonts + vocab_sizes.fontsizes
        hidden1 = 6
        hidden2 = 6
        with tf.variable_scope('conv_decode1'):
            decode1 = self._conv_relu(
                    self.convolution,
                    [3, 3, self.model.filters.conv3, 3*4*hidden1],
                    [1, 1, 1, 1], 'SAME')
            decode1 = tf.reshape(
                    decode1, [-1, 60, 8, 3, 4, 3])
            unstacked = tf.unstack(decode1, axis=3)
            unstacked = [tf.reshape(t, [-1, 60, 32, hidden1])
                         for t in unstacked]
            decode1 = tf.concat(unstacked, axis=2)
            decode1 = tf.reshape(decode1, [-1, 180, 32, hidden1])
        with tf.variable_scope('conv_decode2'):
            decode2 = self._conv_relu(
                    decode1,
                    [2, 2, hidden1, 2*3*hidden2],
                    [1, 1, 1, 1], 'SAME')
            decode2 = tf.reshape(decode2, [-1, 180, 32, 2, 3, hidden2])
            unstacked = tf.unstack(decode2, axis=3)
            unstacked = [tf.reshape(t, [-1, 180, 96, hidden2])
                         for t in unstacked]
            decode2 = tf.concat(unstacked, axis=2)
            decode2 = tf.reshape(decode2, [-1, 360, 96, hidden2])
        with tf.variable_scope('feature_logits'):
            feature_logits = tf.reshape(self._linear_layer(
                tf.reshape(decode2, [-1, hidden2]),
                [hidden2, out_dim]), [self.model.batch_size, -1, out_dim])
            feature_splits = tf.split(
                feature_logits,
                [vocab_sizes.chars, vocab_sizes.fonts, vocab_sizes.fontsizes],
                axis=2
            )
        return feature_splits

    @scope.lazy_load
    def autoencoder_loss(self):
        feature_targets = tf.unstack(tf.reshape(
            self.trim,
            [self.model.batch_size, -1, 3]), axis=2, name='feature_targets')
        cross_entropies = [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=logits)
            for target, logits in zip(feature_targets, self.autoencoder)
        ]
        cross_entropy = tf.add_n(cross_entropies, name='cross_entropy')
        mask = tf.cast(
            tf.sign(feature_targets[0]), dtype=tf.float32, name='mask')

        sparsity = tf.reduce_sum(mask) / tf.cast(tf.size(mask), tf.float32)
        sparsity = (sparsity * self.sparsity_adjustment +
                    (1-self.sparsity_adjustment))
        mask = (1 - sparsity) * mask + sparsity
        cross_entropy = mask * cross_entropy
        cross_entropy = tf.reduce_sum(cross_entropy)
        cross_entropy /= tf.reduce_sum(mask)
        tf.summary.scalar('autoencoder_cross_entropy', cross_entropy,
                          collections=['train', 'test'])
        return cross_entropy

    @scope.lazy_load
    def encoded(self):
        conv_shape = tf.shape(self.convolution)
        # flatten to a sequence of vectors to feed to the encoder
        rnn_inputs = tf.transpose(self.convolution, [0, 2, 1, 3])
        rnn_inputs = tf.reshape(
            self.convolution, [conv_shape[0], -1, self.model.filters.conv3],
            # rnn_inputs, [conv_shape[0], -1, self.model.filters.conv3],
            name='flatten_to_string')
        # inputs_time_major = tf.transpose(rnn_inputs, [1, 0, 2])
        pad_amount = self.sequence_length - tf.shape(rnn_inputs)[1]
        padded_to_length = tf.pad(rnn_inputs,
                                  [[0, 0], [0, pad_amount], [0, 0]],
                                  name='pad')
        # doesn't actually reshape, just provides shape info to future steps
        padded_to_length = tf.reshape(
            padded_to_length,
            [self.model.batch_size, self.sequence_length,
             self.model.filters.conv3])
        return padded_to_length

    @scope.lazy_load
    def decoder(self):
        cell = self._rnn_cell()
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell, self.model.token_vocab_size)
        initial_state = cell.zero_state(self.model.batch_size, dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(
            cell, self.encoded, sequence_length=self.sequence_lengths,
            time_major=False, initial_state=initial_state
        )
        return output

    @scope.lazy_load
    def logits(self):
        decoded_dim = self.model.token_vocab_size
        decoded_flat = tf.reshape(
            self.decoder,
            [-1, decoded_dim])
        return decoded_flat

    @scope.lazy_load_no_scope
    def loss(self):
        with tf.variable_scope('loss') as scope:
            self.loss_scope = scope
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets, logits=self.logits)
            mask = tf.cast(tf.sign(self.targets), dtype=tf.float32)
            cross_entropy = cross_entropy * mask

            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=0)
            cross_entropy /= tf.cast(
                tf.reduce_sum(self.sequence_lengths), tf.float32)
            tf.summary.scalar('cross_entropy', cross_entropy,
                              collections=['train', 'test'])
            return cross_entropy

    @scope.lazy_load_no_scope
    def total_loss(self):
        return (self.total_loss_ratio * self.autoencoder_loss +
                (1 - self.total_loss_ratio) * self.loss)

    @scope.lazy_load_no_scope
    def accuracy(self):
        with tf.variable_scope(self.loss_scope.original_name_scope):
            target_signs = tf.sign(self.targets)
            self.inference = tf.cast(tf.argmax(self.logits, 1), tf.int32)
            correct_prediction = tf.cast(tf.equal(
                self.inference, self.targets, name='correct'), tf.float32)
            self.inference = self.inference * target_signs
            correct_prediction = correct_prediction * tf.cast(
                target_signs, tf.float32)
            accuracy = tf.reduce_sum(correct_prediction)
            accuracy /= tf.cast(
                tf.reduce_sum(self.sequence_lengths), tf.float32)
            tf.summary.scalar('accuracy', accuracy,
                              collections=['train', 'test'])
            return accuracy

    @scope.lazy_load
    def optimize(self):
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
        #                                   self.model.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.model.learning_rate)
        return optimizer.minimize(self.loss, global_step=self.global_step)
        # return optimizer.apply_gradients(
        #     zip(grads, tvars), global_step=self.global_step)

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
            "biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))
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

    def _linear_layer(self, input, shape):
        """
        Constructs tensors W, b and transforms input -> input * W + b
        """
        weights = tf.get_variable(
            'weights', shape, tf.float32,
            initializer=tf.random_normal_initializer()
        )
        self._activation_summary(weights)
        biases = tf.get_variable(
            "biases", shape[1], initializer=tf.constant_initializer(0.0))
        self._activation_summary(biases)
        return tf.matmul(input, weights) + biases

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
