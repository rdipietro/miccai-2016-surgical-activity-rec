from __future__ import division, print_function

import numpy as np


def compute_accuracy(prediction_seq, label_seq):
    """ Compute accuracy, the fraction of correct predictions.

    Args:
        prediction_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.
        label_seq: A 2-D int NumPy array with the same shape.

    Returns:
        A float.
    """

    return np.mean(prediction_seq == label_seq, dtype=np.float)


def compute_edit_distance(prediction_seq, label_seq):
    """ Compute segment-level edit distance.

    First, transform each sequence to the segment level by replacing any
    repeated, adjacent labels with one label. Second, compute the edit distance
    (Levenshtein distance) between the two segment-level sequences.

    Simplified example: pretend each input sequence is only 1-D, with
    `prediction_seq = [1, 3, 2, 2, 3]` and `label_seq = [1, 2, 2, 2, 3]`.
    The segment-level equivalents are `[1, 3, 2, 3]` and `[1, 2, 3]`, resulting
    in an edit distance of 1.

    Args:
        prediction_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.
        label_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.

    Returns:
        A nonnegative integer, the number of operations () to transform the
        segment-level version of `prediction_seq` into the segment-level
        version of `label_seq`.
    """

    def edit_distance(seq1, seq2):

        seq1 = [-1] + list(seq1)
        seq2 = [-1] + list(seq2)

        dist_matrix = np.zeros([len(seq1), len(seq2)], dtype=np.int)
        dist_matrix[:, 0] = np.arange(len(seq1))
        dist_matrix[0, :] = np.arange(len(seq2))

        for i in xrange(1, len(seq1)):
            for j in xrange(1, len(seq2)):
                if seq1[i] == seq2[j]:
                    dist_matrix[i, j] = dist_matrix[i-1, j-1]
                else:
                    operation_dists = [dist_matrix[i-1, j],
                                       dist_matrix[i, j-1],
                                       dist_matrix[i-1, j-1]]
                    dist_matrix[i, j] = np.min(operation_dists) + 1

        return dist_matrix[-1, -1]

    def segment_level(seq):
        segment_level_seq = []
        for label in seq.flatten():
            if len(segment_level_seq) == 0 or segment_level_seq[-1] != label:
                segment_level_seq.append(label)
        return segment_level_seq

    return edit_distance(segment_level(prediction_seq),
                         segment_level(label_seq))


def compute_metrics(prediction_seqs, label_seqs):
    """ Compute metrics averaged over sequences.

    Args:
        prediction_seqs: A list of int NumPy arrays, each with shape
            `[duration, 1]`.
        label_seqs: A list of int NumPy arrays, each with shape
            `[duration, 1]`.

    Returns:
        A tuple,
        mean_accuracy: A float, the average over all sequences of the
            accuracies computed on a per-sequence basis.
        mean_edit_distance: A float, the average over all sequences of
            edit distances computed on a per-sequence basis.
    """

    accuracies = [compute_accuracy(pred_seq, label_seq)
                  for (pred_seq, label_seq) in zip(prediction_seqs, label_seqs)]
    accuracy_mean = np.mean(accuracies, dtype=np.float)
    edit_dists = [compute_edit_distance(pred_seq, label_seq)
                  for (pred_seq, label_seq) in zip(prediction_seqs, label_seqs)]
    edit_dist_mean = np.mean(edit_dists, dtype=np.float)
    return accuracy_mean, edit_dist_mean
