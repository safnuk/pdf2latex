import collections

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

import dataset


FilterSizes = collections.namedtuple('FilterSizes', ['conv1', 'conv2', 'conv3'])


class Network:
    def __init__(
        self,
        model,
        validation_size,
        test_size,
        data_dir='data/',
        log_dir='log/',
    ):
        self.model = model
        self.data_dir = data_dir
        self.log_dir = log_dir + '/' + model.name + '/'
        self.validation_size = validation_size
        self.test_size = test_size
        self.data = dataset.read_datasets(data_dir, validation_size, test_size)
        self.learning_rate = tf.Variable(
            float(model.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * model.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.dropout = tf.placeholder(tf.float32, name='keep_prob')

    def build(self):
        # Feeds for inputs.
        self.feature_input = tf.placeholder(tf.int32, shape=[None, 366, 100, 3])
        self.tokens = tf.placeholder(
            tf.int32, shape=[None, self.data.token_sequence_length],
        )
        embed = self.embedding_layers(self.feature_input)
        conv = self.convolution_layers(embed)
        encode = self.encoder_layer(conv)
        tokens = self.token_embed_layer(self.tokens)
        decode_train = self.decoder_train_layer(encode, tokens)
        self.training_loss = self.loss_layer(decode_train)
        decode_infer = self.decoder_inference_layer(encode, tokens)
        self.inference_loss = self.loss_layer(decode_infer)
        self.optimize = self.train_layer(self.training_loss)

    def feed_dict(self, train=True):
        if train:
            feature_batch, token_batch = self.data.train.next_batch(
                self.model.batch_size)
            keep_prob = self.model.dropout_keep_prob
        else:
            feature_batch, token_batch = self.data.validate.next_batch(
                self.validation_size)
            keep_prob = 1.0
        return {
            self.feature_input: feature_batch,
            self.tokens: token_batch,
            self.dropout: keep_prob,
        }

    def train(self):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.log_dir + 'train', sess.graph)
        test_writer = tf.summary.FileWriter(self.log_dir + 'test')
        tf.global_variables_initializer().run()
        for n in range(self.model.max_steps):
            if n % 10 == 0:
                summary, acc, loss = sess.run(
                    [merged, self.accuracy, self.inference_loss],
                    feed_dict=self.feed_dict(False)
                )
                test_writer.add_summary(summary, n)
                print('Accuracy at step {}: {}  Loss: {}'
                      .format(n, acc, loss))
            else:
                if n % 100 == 99:  # Record execution stats
                    saver.save(sess, self.log_dir + 'checkpoint-',
                               global_step=self.global_step)
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, self.optimize],
                                          feed_dict=self.feed_dict(True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%03d' % n)
                    writer.add_summary(summary, n)
                    print('Adding run metadata for', n)
                else:  # Record a summary
                    summary, _ = sess.run([merged, self.optimize],
                                          feed_dict=self.feed_dict(True))
                    writer.add_summary(summary, n)
        writer.close()
        test_writer.close()

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

    def _embedding_variable(self, vocab_size, embedding_dim):
        initial = tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0)
        return tf.Variable(initial, name='embedding')

    def _kernel_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=5e-2)
        return tf.Variable(initial, name='kernel')

    def _bias_variable(self, shape, constant=0.0):
        initial = tf.constant(constant, shape=shape)
        return tf.Variable(initial, name='bias')

    def embedding_layers(self, input_pdfs):
        """Embed the input data into dense vector representations.

        Args:
            pdfs: Input data returned from :func:`input`.
        Returns:
            Dense vector space representation of the input data.
        """
        vocab_size = self.data.feature_vocab_size
        with tf.name_scope("slice_features"):
            slices = tf.unstack(input_pdfs, axis=3)
        with tf.name_scope('char_embed'):
            embeddings = self._embedding_variable(
                vocab_size.chars, self.model.embedding_dims.chars)
            input_embed = tf.nn.embedding_lookup(embeddings, slices[0])
            self._activation_summary(embeddings)
        with tf.variable_scope('font_embed'):
            embeddings = self._embedding_variable(
                vocab_size.fonts, self.model.embedding_dims.fonts)
            font_embed = tf.nn.embedding_lookup(
                embeddings, slices[1])
            self._activation_summary(embeddings)
        with tf.name_scope('fontsize_embed'):
            embeddings = self._embedding_variable(
                vocab_size.fontsizes, self.model.embedding_dims.fontsizes)
            fontsize_embed = tf.nn.embedding_lookup(
                embeddings, slices[2])
            self._activation_summary(embeddings)
        with tf.name_scope('feature_embed'):
            embedded = tf.concat([input_embed, font_embed, fontsize_embed], 3)
            self._activation_summary(embeddings)
            return embedded

    def convolution_layers(self, embedded):
        """Build the convolution layers of the network.

        Args:
            embedded: Tensors returned from :func:`embedding_layers`.
        Returns:
            Application of three convolution layers.
        """
        with tf.name_scope('conv1'):
            kernel = self._kernel_variable(
                [4, 6, self.model.input_dim, self.model.filters.conv1])
            conv = tf.nn.conv2d(embedded, kernel, [1, 2, 2, 1], padding='VALID')
            biases = self._bias_variable([self.model.filters.conv1], 0.1)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation)
            self._activation_summary(conv1)
        with tf.name_scope('conv2'):
            kernel = self._kernel_variable(
                [3, 6, self.model.filters.conv1, self.model.filters.conv2])
            conv = tf.nn.conv2d(conv1, kernel, [1, 3, 3, 1], padding='VALID')
            biases = self._bias_variable([self.model.filters.conv2], 0.1)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation)
            self._activation_summary(conv2)
        with tf.name_scope('conv3'):
            kernel = self._kernel_variable(
                [3, 3, self.model.filters.conv2, self.model.filters.conv3])
            conv = tf.nn.conv2d(conv2, kernel, [1, 1, 2, 1], padding='SAME')
            biases = self._bias_variable([self.model.filters.conv3], 0.1)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation)
            self._activation_summary(conv3)
        return conv3

    def encoder_layer(self, conv):
        with tf.name_scope('flatten'):
            conv_shape = tf.shape(conv)
            rnn_inputs = tf.reshape(
                conv, [conv_shape[0], -1, self.model.filters.conv3])
        with tf.name_scope('reverse'):
            rnn_reversed = tf.reverse(rnn_inputs, [-1])
        with tf.variable_scope('encoder'):
            cell = self._rnn_cell()
            _, encoded_state = tf.nn.dynamic_rnn(
                cell, rnn_reversed, dtype=tf.float32)
        return encoded_state

    def token_embed_layer(self, token_inputs):
        num_tokens = self.data.token_vocab_size
        with tf.name_scope('slice_off_seq_lengths'):
            token_shape = tf.shape(token_inputs)
            self.target_sequences, lengths = tf.split(
                token_inputs, [token_shape[1] - 1, 1], axis=1)
            self.sequence_lengths = tf.squeeze(lengths)
        with tf.name_scope('embed_tokens'):
            self.token_embedding_matrix = self._embedding_variable(
                num_tokens, self.model.embedding_dims.tokens)
            token_embedding = tf.nn.embedding_lookup(
                self.token_embedding_matrix, self.target_sequences)
            self._activation_summary(token_embedding)
        return token_embedding

    def decoder_train_layer(self, encoded_state, token_embedding):
        num_tokens = self.data.token_vocab_size
        with tf.variable_scope('decoder_train'):
            cell = self._rnn_cell()
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_tokens)
            dynamic_fn_train = seq2seq.simple_decoder_fn_train(encoded_state)
            decoder_outputs, state, context = seq2seq.dynamic_rnn_decoder(
                cell, dynamic_fn_train, token_embedding,
                self.sequence_lengths)
        return decoder_outputs

    def decoder_inference_layer(self, encoded_state, token_embedding):
        num_tokens = self.data.token_vocab_size
        with tf.variable_scope('decoder_test'):
            cell = self._rnn_cell()
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_tokens)
            dynamic_fn_test = seq2seq.simple_decoder_fn_inference(
                None, encoded_state, self.token_embedding_matrix,
                self.data.GO_CODE, self.data.STOP_CODE,
                self.data.token_sequence_length - 2, num_tokens
            )
            decoder_outputs, state, context = seq2seq.dynamic_rnn_decoder(
                cell, dynamic_fn_test)
        return decoder_outputs

    def loss_layer(self, logits):
        with tf.name_scope('loss'):
            with tf.name_scope('trunc_seq'):
                max_time = tf.shape(logits)[1]
                targets = tf.slice(self.target_sequences,
                                   [0, 0], [-1, max_time])
            with tf.name_scope('flatten_logits'):
                token_embedding_dim = tf.shape(logits)[2]
                logits_flat = tf.reshape(logits, [-1, token_embedding_dim])
                targets_flat = tf.reshape(targets, [-1])
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets_flat, logits=logits_flat)
                total_loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross_entropy', total_loss)

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(
                        tf.argmax(logits_flat, 1),
                        tf.cast(targets_flat, tf.int64))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        return total_loss

    def train_layer(self, loss):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(loss, global_step=self.global_step)
            return train_step

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
        batch_size,
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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.max_steps = max_steps
        self.rnn_cell_size = rnn_cell_size
        self.num_rnn_layers = num_rnn_layers
        self.filters = conv_filter_sizes
        self.embedding_dims = embedding_dims
        self.use_lstm = use_lstm
        self.use_rnn_layer_norm = use_rnn_layer_norm
        self.dropout_keep_prob = dropout_keep_prob
        self.input_dim = (embedding_dims.chars + embedding_dims.fonts +
                          embedding_dims.fontsizes)

    @classmethod
    def small(cls):
        return cls('small', 100, 0.01, 1, 100, 64, 1)

    @classmethod
    def medium(cls):
        return cls('medium', 100, 0.001, 1, 2000, 128, 4)

    @classmethod
    def medium_reg(cls):
        return cls('medium-reg', 100, 0.01, 1, 3000, 200, 4,
                   use_lstm=True, use_rnn_layer_norm=True)

if __name__ == '__main__':
    BASE = '/Users/safnu1b/Documents/latex/'
    BASE1 = ''
    # BASE = '/data/safnu1b/latex/'
    network = Network(Model.small(), validation_size=100, test_size=1,
                      data_dir=BASE1 + 'data/',
                      log_dir=BASE + 'data/log/')
    network.build()
    network.train()
