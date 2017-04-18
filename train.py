import os
import shutil

import tensorflow as tf

import model
import network3 as net


def train(model, logdir, clear_old_logs=True):
    logdir = logdir + model.name + '/'
    if clear_old_logs:
        if os.path.exists(logdir):
            shutil.rmtree(logdir)

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
    train_summaries = tf.summary.merge_all('train')

    sv = tf.train.Supervisor(logdir=logdir, summary_op=None)
    with sv.managed_session() as sess:
        infer_writer = tf.summary.FileWriter(logdir + 'infer/')

        step = 0
        while step < network.model.max_steps:
            if sv.should_stop():
                break
            if step % 100 == 0:
                epoch = network.model.data.validate.epochs_completed
                runs = 0
                total_loss = 0
                total_acc = 0
                while epoch == network.model.data.validate.epochs_completed:
                    runs += 1
                    loss, acc, step = sess.run(
                        [network.loss, network.accuracy,
                         network.global_step],
                        feed_dict=feed_dict(train=False))
                    total_loss += loss
                    total_acc += acc
                loss = total_loss / runs
                acc = total_acc / runs
                loss_summary = tf.Summary()
                loss_summary.value.add(
                    tag="loss/cross_entropy", simple_value=loss)
                acc_summary = tf.Summary()
                acc_summary.value.add(tag="loss/accuracy", simple_value=acc)
                infer_writer.add_summary(loss_summary, global_step=step)
                infer_writer.add_summary(acc_summary, global_step=step)
                print(
                    'Round {} Validation loss: {} Accuracy: {}'
                    .format(step, loss, acc))
            if step % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                loss, acc, s, step, _ = sess.run(
                    [network.loss, network.accuracy, train_summaries,
                        network.global_step, network.optimize],
                    feed_dict=feed_dict(train=True),
                    options=run_options,
                    run_metadata=run_metadata)
                print('Adding run metadata for', step)
                sv.summary_writer.add_run_metadata(
                    run_metadata, 'step%03d' % step)
                sv.summary_computed(sess, s)
            else:
                loss, acc, s, step, _ = sess.run(
                    [network.loss, network.accuracy, train_summaries,
                        network.global_step, network.optimize],
                    feed_dict=feed_dict(train=True))
                sv.summary_computed(sess, s)
                if step % 10 == 0:
                    print("Step {} Train loss: {} Accuracy: {}"
                          .format(step, loss, acc))

    infer_writer.close()

if __name__ == '__main__':
    # BASE = '/data/safnu1b/latex/'
    BASE = '/Users/safnu1b/Documents/latex/'
    BASE1 = ''
    datadir = BASE1 + 'data/'
    logdir = BASE + 'log/'
    model = model.Model.small(datadir)
    train(model, logdir, True)
