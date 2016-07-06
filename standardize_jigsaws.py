from __future__ import division, print_function

import os
import argparse
import itertools

import numpy as np
import cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data

DATASET_NAME = 'JIGSAWS'
ORIG_CLASS_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
NEW_CLASS_IDS = range(len(ORIG_CLASS_IDS))
CLASSES = ['G%d' % id for id in ORIG_CLASS_IDS]
NUM_CLASSES = len(CLASSES)

# Standard JIGSAWS experimental setup. In particular, it's the only
# recognition setup that exists, corresponding to
# JIGSAWS/Experimental/Suturing/unBalanced/GestureRecognition/UserOut
# (User H's 2nd trial is left out because no video was available for labeling.)
USER_TO_TRIALS = {
    'B': [1, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 5],
    'D': [1, 2, 3, 4, 5],
    'E': [1, 2, 3, 4, 5],
    'F': [1, 2, 3, 4, 5],
    'G': [1, 2, 3, 4, 5],
    'H': [1,    3, 4, 5],
    'I': [1, 2, 3, 4, 5]
}

ALL_USERS = sorted(USER_TO_TRIALS.keys())

KINEMATICS_USECOLS = [c-1 for c in [39, 40, 41, 51, 52, 53, 57,
                                    58, 59, 60, 70, 71, 72, 76]]
KINEMATICS_COL_NAMES = ['pos_x', 'pos_y', 'pos_z', 'vel_x',
                        'vel_y', 'vel_z', 'gripper']*2

LABELS_USECOLS = [0, 1, 2]
LABELS_COL_NAMES = ['start_frame', 'end_frame', 'string_label']
LABELS_CONVERTERS = {2: lambda string_label: int(string_label.replace('G', ''))}

STANDARDIZED_COL_NAMES = KINEMATICS_COL_NAMES + ['label']


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
                                we'll create inside data_dir.''')

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def get_trial_name(user, trial):
    """ Form a trial name that matches standard JIGSAWS filenames.

    Args:
        user: A string.
        trial: An integer.

    Returns:
        A string.
    """
    return 'Suturing_%s%03d' % (user, trial)


def load_kinematics(data_dir, trial_name):
    """ Load kinematics data.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis.
    """

    kinematics_dir = os.path.join(data_dir, 'kinematics', 'AllGestures')
    kinematics_path = os.path.join(kinematics_dir, trial_name + ".txt")
    data = np.loadtxt(kinematics_path, dtype=np.float,
                      usecols=KINEMATICS_USECOLS)
    return data


def load_kinematics_and_labels(data_dir, trial_name):
    """ Load kinematics data and labels.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended
        as a new column to the raw kinematics data (and are therefore
        represented as floats).
    """

    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    kinematics_data = load_kinematics(data_dir, trial_name)
    raw_labels_data = np.genfromtxt(labels_path, dtype=np.int,
                                    converters=LABELS_CONVERTERS,
                                    usecols=LABELS_USECOLS)
    frames = np.arange(1, kinematics_data.shape[0]+1, dtype=np.int)
    labels = np.zeros(frames.shape, dtype=np.int)
    for start, end, label in raw_labels_data:
        mask = (frames >= start) & (frames <= end)
        labels[mask] = label
    labels_data = labels.reshape(-1, 1)

    data = np.concatenate([kinematics_data, labels_data], axis=1)
    labeled_data_only_mask = labels_data.flatten() != 0

    return data[labeled_data_only_mask, :]


def load_kinematics_and_new_labels(data_dir, trial_name):
    """ Load kinematics data and standardized labels.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended as
        a new column and are converted from arbitrary labels (e.g., 1, 3, 5)
        to ordered, nonnegative integers (e.g., 0, 1, 2).
    """
    data = load_kinematics_and_labels(data_dir, trial_name)
    for orig, new in zip(ORIG_CLASS_IDS, NEW_CLASS_IDS):
        mask = data[:, -1] == orig
        data[mask, -1] = new
    return data


def downsample(data, factor=6):
    """ Downsample a data matrix.

    Args:
        data: A 2-D NumPy array with time on the first axis.
        factor: The factor by which we'll downsample.

    Returns:
        A 2-D NumPy array.
    """
    return data[::factor, :]


def main():
    """ Create a standardized data file from raw data. """

    args = define_and_process_args()

    print('Standardizing JIGSAWS..')
    print()

    print('%d classes:' % NUM_CLASSES)
    print(CLASSES)
    print()

    user_to_trial_names = {}
    for user, trials in USER_TO_TRIALS.items():
        user_to_trial_names[user] = [get_trial_name(user, trial)
                                     for trial in trials]
    print('Users and corresponding trial names:')
    for user in ALL_USERS:
        print(user, '   ', user_to_trial_names[user])
    print()


    all_trial_names = sorted(list(
        itertools.chain(*user_to_trial_names.values())
    ))
    print('All trial names, sorted:')
    print(all_trial_names)
    print()

    # Original data is at 30 Hz.
    all_data = {trial_name: downsample(
                    load_kinematics_and_new_labels(args.data_dir, trial_name),
                    factor=6)
                for trial_name in all_trial_names}
    print('Downsampled to 5 Hz.')
    print()

    fig, ax_list = plt.subplots(nrows=len(all_data), ncols=1,
                                sharex=True, figsize=(15, 50))
    for ax, (trial_name, data_mat) in zip(ax_list, sorted(all_data.items())):
        plt.sca(ax)
        data.plot_label_seq(data_mat[:, -1:], NUM_CLASSES, 0)
        plt.title(trial_name)
    plt.tight_layout()
    vis_path = os.path.join(args.data_dir, 'standardized_data_labels.png')
    plt.savefig(vis_path)
    plt.close(fig)
    print('Saved label visualization to %s.' % vis_path)
    print()

    export_dict = dict(
        dataset_name=DATASET_NAME, classes=CLASSES, num_classes=NUM_CLASSES,
        col_names=STANDARDIZED_COL_NAMES, all_users=ALL_USERS,
        user_to_trial_names=user_to_trial_names,
        all_trial_names=all_trial_names, all_data=all_data)
    standardized_data_path = os.path.join(args.data_dir, args.data_filename)
    with open(standardized_data_path, 'w') as f:
        cPickle.dump(export_dict, f)
    print('Saved standardized data file %s.' % standardized_data_path)
    print()


if __name__ == '__main__':
    main()
