from __future__ import division, print_function

import itertools
import cPickle

import numpy as np
import matplotlib.pyplot as plt


class Dataset(object):

    def __init__(self, pkl_path):
        """ Create a Dataset object from a standardized Pickle file.

        JIGSAWS and MISTIC contain similar underlying data, namely kinematics
        as input and surgical activity as output. This class loads a
        standardized Pickle file that can contain data for either dataset. See
        the properties below for a description of what the file must contain.

        Args:
            pkl_path: A string. A path to the standardized Pickle file.
        """

        with open(pkl_path, 'r') as f:
            self.pkl_dict = cPickle.load(f)

        assert all(seq.shape[1] - 1 == self.input_size
                   for seq in self.all_data.values())

    def get_seqs_by_user(self, user):
        """ Get a list of sequences corresponding to a user.

        Args:
            user: A string.

        Returns:
            A list of sequences corresponding to `user`.
        """

        trial_names = sorted(self.user_to_trial_names[user])
        seqs = [self.all_data[trial_name] for trial_name in trial_names]
        return seqs

    def get_splits(self, test_users):
        """ Get all sequences, split into a training set and a testing set.

        Args:
            test_users: A list of strings.

        Returns:
            A tuple,
            train_seqs: A list of train sequences.
            test_seqs: A list of test sequences.
        """

        train_users = [user for user in self.all_users
                       if user not in test_users]
        train_seqs = list(itertools.chain(*[self.get_seqs_by_user(user)
                                            for user in train_users]))
        test_seqs = list(itertools.chain(*[self.get_seqs_by_user(user)
                                           for user in test_users]))

        # Sanity check
        def seqs_are_same(seq1, seq2):
            same_shape = seq1.shape == seq2.shape
            return same_shape and np.allclose(seq1, seq2, rtol=1e-3, atol=1e-3)
        for test_seq in test_seqs:
            assert any([seqs_are_same(test_seq, test_seq_)
                        for test_seq_ in test_seqs])
            assert not any([seqs_are_same(test_seq, train_seq)
                            for train_seq in train_seqs])

        return train_seqs, test_seqs

    @property
    def dataset_name(self):
        """ A string: the dataset name. """
        return self.pkl_dict['dataset_name']

    @property
    def classes(self):
        """ A list of strings: the class names. """
        return self.pkl_dict['classes']

    @property
    def num_classes(self):
        """ An integer: the number of classes. """
        return self.pkl_dict['num_classes']

    @property
    def all_users(self):
        """ A list of strings, each representing a user. """
        return self.pkl_dict['all_users']

    @property
    def all_trial_names(self):
        """ A list of strings: all trial names over all users. """
        return self.pkl_dict['all_trial_names']

    @property
    def user_to_trial_names(self):
        """ A dictionary mapping users to trial-name lists. """
        return self.pkl_dict['user_to_trial_names']

    @property
    def all_data(self):
        """ A dictionary mapping trial names to NumPy arrays. Each NumPy
            array has shape `[duration, input_size+1]`, with the last
            column being class labels. """
        return self.pkl_dict['all_data']

    @property
    def col_names(self):
        """ A list of strings: the column names for each data column. """
        return self.pkl_dict['col_names']

    @property
    def input_size(self):
        """ An integer: the number of inputs per time step. """
        return self.all_data.values()[0].shape[1] - 1


def normalize_seq(seq):
    """ Normalize a sequence by centering/scaling columns.

    Args:
        seq: A 2-D NumPy array with shape `[duration, size]`.

    Returns:
        A 2-D NumPy array with the same shape, but with all columns
        having mean 0 and standard deviation 1.
    """

    mu = seq.mean(axis=0, keepdims=True)
    sigma = seq.std(axis=0, keepdims=True)
    normalized_seq = (seq - mu) / sigma
    return normalized_seq


def prepare_raw_seq(seq):
    """ Prepare a raw sequence for training/testing.

    This function a) splits a raw sequence into input and label sequences; b)
    prepares a reset sequence (for handling RNN state resets); and c)
    normalizes each input sequence.

    Args:
        seq: A 2-D NumPy array with shape `[duration, num_inputs + 1]`.
            The last column stores labels.

    Returns:
        A tuple,
        input_seq: A 2-D float32 NumPy array with shape
            `[duration, num_inputs]`. A normalized input sequence.
        reset_seq: A 2-D bool NumPy array with shape `[duration, 1]`.
        label_seq: A 2-D int NumPy array with shape `[duration, 1]`.
    """

    input_seq = seq[:, :-1].astype(np.float)
    input_seq = normalize_seq(input_seq).astype(np.float32)
    duration = input_seq.shape[0]
    reset_seq = np.eye(1, duration, dtype=np.bool).T
    label_seq = seq[:, -1:].astype(np.int)
    return input_seq, reset_seq, label_seq


def seq_ind_generator(num_seqs, shuffle=True):
    """ A sequence-index generator.

    Args:
        num_seqs: An integer. The number of sequences we'll be indexing.
        shuffle: A boolean. If true, randomly shuffle indices epoch by epoch.

    Yields:
        An integer in `[0, num_seqs)`.
    """

    seq_inds = range(num_seqs)
    while True:
        if shuffle:
            np.random.shuffle(seq_inds)
        for seq_ind in seq_inds:
            yield seq_ind


def sweep_generator(seq_list_list, batch_size, shuffle=False, num_sweeps=None):
    """ Generate sweeps.

    Let's define a sweep as a collection of `batch_size` sequences that
    continue together through time until all sequences in the batch have been
    exhausted. Short sequences grow by being wrapped in time.

    Simplified example: pretend sequences are 1-D arrays, and that we have
    `seq_list = [[1, 0], [1, 0, 0]]`. Then
    `sweep_generator([seq_list], 3, shuffle=False)` would yield
    `[ [[1, 0, 1], [1, 0, 0], [1, 0, 1]] ]`.

    Args:
        seq_list_list: A list of sequence lists. The sequences in
            `seq_list_list[0]` should correspond to the sequences in
            `seq_list_list[1]`, in `seq_list_list[2]`, etc. Their durations
            should be the same, but data types can differ. All sequences
            should be 2-D and have time running along axis 0.
        batch_size: An integer. The number of sequences in a batch.
        shuffle: A boolean. If true, shuffle sequences epoch by epoch as we
            populate sweeps.
        num_sweeps: An integer. The number of sweeps to visit before the
            generator is exhaused. If None, generate sweeps forever.

    Yields:
        A list with the same length as `seq_list_list`. This contains a sweep
        for the 1st seq list, a sweep for the 2nd seq list, etc., each sweep
        being a NumPy array with shape `[batch_size, duration, ?]`.
    """

    if num_sweeps is None:
        num_sweeps = np.inf

    seq_durations = [len(seq) for seq in seq_list_list[0]]
    num_seqs = len(seq_list_list[0])
    seq_ind_gen = seq_ind_generator(num_seqs, shuffle=shuffle)

    for seq_list in seq_list_list:
        assert len(seq_list) == num_seqs
        assert [len(seq) for seq in seq_list] == seq_durations

    num_sweeps_visited = 0
    while num_sweeps_visited < num_sweeps:

        new_seq_ind = [seq_ind_gen.next() for _ in xrange(batch_size)]
        new_seq_durations = [seq_durations[i] for i in new_seq_ind]
        longest_duration = np.max(new_seq_durations)
        pad = lambda seq: np.pad(seq, [[0, longest_duration-len(seq)], [0, 0]],
                                 mode='wrap')

        new_sweep_list = []
        for seq_list in seq_list_list:
            new_seq_list = [seq_list[i] for i in new_seq_ind]
            new_sweep = np.asarray([pad(seq) for seq in new_seq_list])
            new_sweep_list.append(new_sweep)

        yield new_sweep_list
        num_sweeps_visited += 1


def plot_label_seq(label_seq, num_classes, y_value):
    """ Plot a label sequence.

    The sequence will be shown using a horizontal colored line, with colors
    corresponding to classes.

    Args:
        label_seq: An int NumPy array with shape `[duration, 1]`.
        num_classes: An integer.
        y_value: A float. The y value at which the horizontal line will sit.
    """

    label_seq = label_seq.flatten()
    x = np.arange(0, label_seq.size)
    y = y_value*np.ones(label_seq.size)
    plt.scatter(x, y, c=label_seq, marker='|', lw=2, vmin=0, vmax=num_classes)


def visualize_predictions(prediction_seqs, label_seqs, num_classes,
                          fig_width=6.5, fig_height_per_seq=0.5):
    """ Visualize predictions vs. ground truth.

    Args:
        prediction_seqs: A list of int NumPy arrays, each with shape
            `[duration, 1]`.
        label_seqs: A list of int NumPy arrays, each with shape `[duration, 1]`.
        num_classes: An integer.
        fig_width: A float. Figure width (inches).
        fig_height_per_seq: A float. Figure height per sequence (inches).

    Returns:
        A tuple of the created figure, axes.
    """

    num_seqs = len(label_seqs)
    max_seq_length = max([seq.shape[0] for seq in label_seqs])
    figsize = (fig_width, num_seqs*fig_height_per_seq)
    fig, axes = plt.subplots(nrows=num_seqs, ncols=1,
                             sharex=True, figsize=figsize)

    for pred_seq, label_seq, ax in zip(prediction_seqs, label_seqs, axes):
        plt.sca(ax)
        plot_label_seq(label_seq, num_classes, 1)
        plot_label_seq(pred_seq, num_classes, -1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.xlim(0, max_seq_length)
        plt.ylim(-2.75, 2.75)
        plt.tight_layout()

    return fig, axes


def one_hot(labels, num_classes):
    """ Convert labels to one-hot encodings.

    Args:
        labels: A NumPy array of nonnegative labels.

    Returns:
        A NumPy array with shape `labels.shape + [num_classes]`. That is,
        the same shape is retained, but one axis is added for the one-hot
        encodings.
    """

    encoding_matrix = np.zeros([labels.size, num_classes])
    encoding_matrix[range(labels.size), labels.flatten()] = 1
    encodings = encoding_matrix.reshape(list(labels.shape) + [num_classes])
    return encodings
