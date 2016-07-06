from __future__ import division, print_function

import os
import time
import argparse
import shutil
import cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import data
import models
import optimizers
import metrics


def define_and_process_args():
    """ Define and process command-line arguments.

    Returns:
        A Namespace with arguments as attributes.
    """

    description = main.__doc__
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=formatter_class)

    parser.add_argument('--data_dir', default='~/Data/JIGSAWS/Suturing',
                        help='Data directory.')
    parser.add_argument('--data_filename', default='standardized_data.pkl',
                        help='''The name of the standardized-data pkl file that
                                resides in data_dir.''')
    parser.add_argument('--test_users', default='B',
                        help='''A string of the users that make up the test set,
                                with users separated by spaces.''')

    parser.add_argument('--model_type', default='BidirectionalLSTM',
                        help='''The model type, either BidirectionalLSTM,
                                ForwardLSTM, or ReverseLSTM.''')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument('--hidden_layer_size', type=int, default=1024,
                        help='The number of hidden units per layer.')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='''The fraction of inputs to keep whenever dropout
                                is applied.''')

    parser.add_argument('--batch_size', type=int, default=5,
                        help='The number of sequences in a batch/sweep.')
    parser.add_argument('--num_train_sweeps', type=int, default=600,
                        help='''The number of training sweeps. A sweep
                                is a collection of batch_size sequences that
                                continue together throughout time until all
                                sequences in the batch are exhausted. Short
                                sequences grow by being wrapped around in
                                time.''')
    parser.add_argument('--initial_learning_rate', type=float, default=1.0,
                         help='The initial learning rate.')
    parser.add_argument('--num_initial_sweeps', type=int, default=300,
                        help='''The number of initial sweeps before the
                                learning rate begins to decay.''')
    parser.add_argument('--num_sweeps_per_decay', type=int, default=50,
                        help='''The number of sweeps per learning-rate decay,
                                once decaying begins.''')
    parser.add_argument('--decay_factor', type=float, default=0.5,
                        help='The multiplicative learning-rate-decay factor.')
    parser.add_argument('--max_global_grad_norm', type=float, default=1.0,
                        help='''The global norm is the norm of all gradients
                                when concatenated together. If this global norm
                                exceeds max_global_grad_norm, then all gradients
                                are rescaled so that the global norm becomes
                                max_global_grad_norm.''')

    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='''All weights will be initialized using a
                                uniform distribution over
                                [-init_scale, init_scale].''')
    parser.add_argument('--num_sweeps_per_summary', type=int, default=7,
                        help='''The number of sweeps between summaries. Note:
                                7 sweeps with 5 sequences per sweep corresponds
                                to (more than) 35 visited sequences, which is
                                approximately 1 epoch.''')
    parser.add_argument('--num_sweeps_per_save', type=int, default=7,
                        help='The number of sweeps between saves.')

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.test_users = args.test_users.split(' ')
    return args


def get_log_dir(args):
    """ Form a convenient log directory that summarizes arguments.

    Args:
        args: An object containing processed arguments as attributes.

    Returns:
        A string. The full path to log directory.
    """

    params_str = '_'.join([args.model_type,
                           '%d' % args.num_layers,
                           '%03d' % args.hidden_layer_size,
                           '%02d' % args.batch_size,
                           '%.2f' % args.dropout_keep_prob,
                           '%.4f' % args.initial_learning_rate])

    test_users_str = '_'.join(args.test_users)

    return os.path.join(args.data_dir, 'logs', params_str, test_users_str)


def train(sess, model, optimizer, log_dir, batch_size, num_sweeps_per_summary,
          num_sweeps_per_save, train_input_seqs, train_reset_seqs,
          train_label_seqs, test_input_seqs, test_reset_seqs, test_label_seqs):
    """ Train a model and export summaries.

    `log_dir` will be *replaced* if it already exists, so it certainly
    shouldn't be anything generic like `/home/user`.

    Args:
        sess: A TensorFlow `Session`.
        model: An `LSTMModel`.
        optimizer: An `Optimizer`.
        log_dir: A string. The full path to the log directory.
        batch_size: An integer. The number of sequences in a batch.
        num_sweeps_per_summary: An integer. The number of sweeps between
            summaries.
        num_sweeps_per_save: An integer. The number of sweeps between saves.
        train_input_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, input_size]`.
        train_reset_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        train_label_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        test_input_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, input_size]`.
        test_reset_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        test_label_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
    """

    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    update_train_loss_ema = ema.apply([model.loss])
    train_loss_ema = ema.average(model.loss)
    tf.scalar_summary('train_loss_ema', train_loss_ema)

    train_accuracy = tf.placeholder(tf.float32, name='train_accuracy')
    train_edit_dist = tf.placeholder(tf.float32, name='train_edit_dist')
    test_accuracy = tf.placeholder(tf.float32, name='test_accuracy')
    test_edit_dist = tf.placeholder(tf.float32, name='test_edit_dist')
    values = [train_accuracy, train_edit_dist, test_accuracy, test_edit_dist]
    tags = [value.op.name for value in values]
    tf.scalar_summary('learning_rate', optimizer.learning_rate)
    tf.scalar_summary(tags, tf.pack(values))

    summary_op = tf.merge_all_summaries()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    summary_writer = tf.train.SummaryWriter(logdir=log_dir, graph=sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    num_sweeps_visited = 0
    start_time = time.time()
    train_gen = data.sweep_generator(
        [train_input_seqs, train_reset_seqs, train_label_seqs],
        batch_size=batch_size, shuffle=True, num_sweeps=None)
    while num_sweeps_visited <= optimizer.num_train_sweeps:

        if num_sweeps_visited % num_sweeps_per_summary == 0:

            train_prediction_seqs = models.predict(
                sess, model, train_input_seqs, train_reset_seqs)
            train_accuracy_, train_edit_dist_ = metrics.compute_metrics(
                train_prediction_seqs, train_label_seqs)
            test_prediction_seqs = models.predict(
                sess, model, test_input_seqs, test_reset_seqs)
            test_accuracy_, test_edit_dist_ = metrics.compute_metrics(
                test_prediction_seqs, test_label_seqs)
            summary = sess.run(summary_op,
                               feed_dict={train_accuracy: train_accuracy_,
                                          train_edit_dist: train_edit_dist_,
                                          test_accuracy: test_accuracy_,
                                          test_edit_dist: test_edit_dist_})
            summary_writer.add_summary(summary, global_step=num_sweeps_visited)

            status_path = os.path.join(log_dir, 'status.txt')
            with open(status_path, 'w') as f:
                line = '%05.1f      ' % ((time.time() - start_time)/60)
                line += '%04d      ' % num_sweeps_visited
                line += '%.6f  %08.3f     ' % (train_accuracy_,
                                               train_edit_dist_)
                line += '%.6f  %08.3f     ' % (test_accuracy_,
                                               test_edit_dist_)
                print(line, file=f)

            label_path = os.path.join(log_dir, 'test_label_seqs.pkl')
            with open(label_path, 'w') as f:
                cPickle.dump(test_label_seqs, f)

            pred_path = os.path.join(log_dir, 'test_prediction_seqs.pkl')
            with open(pred_path, 'w') as f:
                cPickle.dump(test_prediction_seqs, f)

            vis_filename = 'test_visualizations_%06d.png' % num_sweeps_visited
            vis_path = os.path.join(log_dir, vis_filename)
            fig, axes = data.visualize_predictions(test_prediction_seqs,
                                                   test_label_seqs,
                                                   model.target_size)
            axes[0].set_title(line)
            plt.tight_layout()
            plt.savefig(vis_path)
            plt.close(fig)

        if num_sweeps_visited % num_sweeps_per_save == 0:
            saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

        train_inputs, train_resets, train_labels = train_gen.next()
        # We squeeze here because otherwise the targets would have shape
        # [batch_size, duration, 1, num_classes].
        train_targets = data.one_hot(train_labels, model.target_size)
        train_targets = train_targets.squeeze(axis=2)

        _, _, num_sweeps_visited = sess.run(
            [optimizer.optimize_op,
             update_train_loss_ema,
             optimizer.num_sweeps_visited],
            feed_dict={model.inputs: train_inputs,
                       model.resets: train_resets,
                       model.targets: train_targets,
                       model.training: True})


def main():
    """ Run training and export summaries to data_dir/logs for a single test
    setup and a single set of parameters. Summaries include a) TensorBoard
    summaries, b) the latest train/test accuracies and raw edit distances
    (status.txt), c) the latest test predictions along with test ground-truth
    labels (test_label_seqs.pkl, test_prediction_seqs.pkl), d) visualizations
    as training progresses (test_visualizations_######.png)."""

    args = define_and_process_args()
    print('\n', 'ARGUMENTS', '\n\n', args, '\n')

    log_dir = get_log_dir(args)
    print('\n', 'LOG DIRECTORY', '\n\n', log_dir, '\n')

    standardized_data_path = os.path.join(args.data_dir, args.data_filename)
    if not os.path.exists(standardized_data_path):
        message = '%s does not exist.' % standardized_data_path
        raise ValueError(message)

    dataset = data.Dataset(standardized_data_path)
    train_raw_seqs, test_raw_seqs = dataset.get_splits(args.test_users)
    train_triplets = [data.prepare_raw_seq(seq) for seq in train_raw_seqs]
    test_triplets = [data.prepare_raw_seq(seq) for seq in test_raw_seqs]

    train_input_seqs, train_reset_seqs, train_label_seqs = zip(*train_triplets)
    test_input_seqs, test_reset_seqs, test_label_seqs = zip(*test_triplets)

    Model = eval('models.' + args.model_type + 'Model')
    input_size = dataset.input_size
    target_size = dataset.num_classes

    # This is just to satisfy a low-CPU requirement on our cluster
    # when using GPUs.
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        config = tf.ConfigProto(intra_op_parallelism_threads=2,
                                inter_op_parallelism_threads=2)
    else:
        config = None

    with tf.Session(config=config) as sess:
        model = Model(input_size, target_size, args.num_layers,
                      args.hidden_layer_size, args.init_scale,
                      args.dropout_keep_prob)
        optimizer = optimizers.Optimizer(
            model.loss, args.num_train_sweeps, args.initial_learning_rate,
            args.num_initial_sweeps, args.num_sweeps_per_decay,
            args.decay_factor, args.max_global_grad_norm)
        train(sess, model, optimizer, log_dir, args.batch_size,
              args.num_sweeps_per_summary, args.num_sweeps_per_save,
              train_input_seqs, train_reset_seqs, train_label_seqs,
              test_input_seqs, test_reset_seqs, test_label_seqs)


if __name__ == '__main__':
    main()
