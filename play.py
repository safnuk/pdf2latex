import tensorflow as tf

import model
import network4 as net


def train(model):
    def feed_dict(train=True):
        if train:
            batch = network.model.data.train.next_batch(
                network.model.batch_size)
        else:
            batch = network.model.data.validate.next_batch(
                network.model.batch_size)
        return {
            network.input.pdf: batch.pdf,
            network.input.token: batch.token,
        }

    network = net.Network(model)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(network.loss, feed_dict=feed_dict())
    print(result)
    result = sess.run(tf.shape(network.loss), feed_dict=feed_dict())
    print(result)

if __name__ == '__main__':
    datadir = 'data/'
    model = model.Model.small(datadir)
    train(model)
