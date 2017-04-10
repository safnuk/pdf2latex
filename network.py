import re

import numpy as np
import tensorflow as tf

import dataset

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


VOCAB_SIZE = {'chars': 10, 'fonts': 5, 'fontsizes': 5}
EMBEDDING_DIMS = {'chars': 3, 'fonts': 2, 'fontsizes': 1}


def _embedding_dim():
    return sum(EMBEDDING_DIMS.values())


def embedding_layers(input_pdfs):
    """Embed the input data into dense vector representations.

    Args:
        pdfs: Input data returned from :func:`input`.
    Returns:
        Dense vector space representation of the input data.
    """
    slices = tf.unstack(input_pdfs, axis=3)
    with tf.variable_scope('char_embed') as scope:
        embeddings = _variable_on_cpu(
            'embeddings',
            [VOCAB_SIZE['chars'], EMBEDDING_DIMS['chars']],
            tf.random_uniform_initializer(-1.0, 1.0)
        )
        input_embed = tf.nn.embedding_lookup(embeddings,
                                             slices[0], name=scope.name)
    with tf.variable_scope('font_embed') as scope:
        embeddings = _variable_on_cpu(
            'embeddings',
            [VOCAB_SIZE['fonts'], EMBEDDING_DIMS['fonts']],
            tf.random_uniform_initializer(-1.0, 1.0)
        )
        font_embed = tf.nn.embedding_lookup(embeddings,
                                            slices[1], name=scope.name)
    with tf.variable_scope('fontsize_embed') as scope:
        embeddings = _variable_on_cpu(
            'embeddings',
            [VOCAB_SIZE['fontsizes'], EMBEDDING_DIMS['fontsizes']],
            tf.random_uniform_initializer(-1.0, 1.0)
        )
        fontsize_embed = tf.nn.embedding_lookup(embeddings,
                                                slices[2], name=scope.name)

    embedded = tf.concat([input_embed, font_embed, fontsize_embed], 3)
    return embedded


def convolution_layers(embedded):
    """Build the convolution layers of the network.

    Args:
        embedded: Tensors returned from :func:`embedding_layers`.
    Returns:
        Application of three convolution layers.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPUs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        conv1_num_filters = 5
        kernel = _variable_with_weight_decay(
            'weights', shape=[4, 6, _embedding_dim(), conv1_num_filters],
            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(embedded, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [conv1_num_filters],
                                  tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # conv2
    with tf.variable_scope('conv2') as scope:
        conv2_num_filters = 3
        kernel = _variable_with_weight_decay(
            'weights', shape=[3, 6, conv1_num_filters, conv2_num_filters],
            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 3, 3, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [conv2_num_filters],
                                  tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # conv3
    with tf.variable_scope('conv3') as scope:
        conv3_num_filters = 5
        kernel = _variable_with_weight_decay(
            'weights', shape=[3, 3, conv2_num_filters, conv3_num_filters],
            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [conv3_num_filters],
                                  tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    return conv3


def _dummy():
    pass


if __name__ == '__main__':
    data = dataset.read_datasets('data/', validation_size=100)
    batch, _ = data.train.next_batch(100)
    input_shape = batch.shape
    test_data = np.random.randint(0, 5, size=input_shape, dtype='int32')
    sess = tf.InteractiveSession()
    input_pdfs = tf.placeholder(tf.int32, shape=input_shape)
    e = embedding_layers(input_pdfs)
    c = convolution_layers(e)
    tf.global_variables_initializer().run()
    shape = tf.shape(c)
    print(shape.eval(feed_dict={input_pdfs: test_data}))
