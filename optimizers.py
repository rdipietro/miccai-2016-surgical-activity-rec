from __future__ import division, print_function

import numpy as np
import tensorflow as tf


def piecewise_constant(x, boundaries, values):
    """ Piecewise constant function.

    Arguments:
        x: A 0-D Tensor.
        boundaries: A 1-D NumPy array with strictly increasing entries.
        values: A 1-D NumPy array that specifies the values for the intervals
            defined by `boundaries`. (It should therefore have one more entry
            than `boundaries`.)

    Returns: A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
        `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ..., and
        values[-1] when `x > boundaries[-1]`.
    """

    pred_fn_pairs = {}
    pred_fn_pairs[x <= boundaries[0]] = lambda: tf.constant(values[0])
    pred_fn_pairs[x > boundaries[-1]] = lambda: tf.constant(values[-1])
    for lower, upper, value in zip(boundaries[:-1],
                                   boundaries[1:],
                                   values[1:-1]):
        # We need to bind value here; can do this with lambda value=value: ...
        pred = (x > lower) & (x <= upper)
        pred_fn_pairs[pred] = lambda value=value: tf.constant(value)

    return tf.case(pred_fn_pairs, lambda: tf.constant(values[0]),
                   exclusive=True)


class Optimizer(object):

    def __init__(self, loss, num_train_sweeps, initial_learning_rate,
                 num_initial_sweeps, num_sweeps_per_decay, decay_factor,
                 max_global_grad_norm):
        """ Create an optimizer.

        If the global norm over all gradients is greater than
        `max_global_grad_norm`, then scale all gradients so that the global
        norm becomes `max_global_grad_norm`. Then update using vanilla SGD with
        a staircase learning-rate schedule which a) maintains the initial
        learning rate for `num_initial_sweeps` sweeps and then b) decays
        the learning rate by a factor of `decay_factor` every
        `num_sweeps_per_decay` sweeps.

        Args:
            loss: A 0-D float32 Tensor.
            num_train_sweeps: An integer.
            initial_learning_rate: A float.
            num_initial_sweeps: An integer.
            num_sweeps_per_decay: An integer.
            decay_factor: A float.
            max_global_grad_norm: A float.
        """

        self.loss = loss
        self.initial_learning_rate = initial_learning_rate
        self.num_initial_sweeps = num_initial_sweeps
        self.num_sweeps_per_decay = num_sweeps_per_decay
        self.decay_factor = decay_factor
        self.max_global_grad_norm = max_global_grad_norm
        self.num_train_sweeps = num_train_sweeps

        self._trainables = tf.trainable_variables()
        self._raw_grads = tf.gradients(loss, self.trainables)
        self._scaled_grads, _ = tf.clip_by_global_norm(
            self.raw_grads, clip_norm=self.max_global_grad_norm)

        self._num_sweeps_visited = tf.Variable(0, trainable=False,
                                               name='num_sweeps_visited',
                                               dtype=tf.int32)
        boundaries = np.arange(self.num_initial_sweeps, self.num_train_sweeps,
                               self.num_sweeps_per_decay, dtype=np.int32)
        values = [self.initial_learning_rate * self.decay_factor**i
                  for i in xrange(len(boundaries)+1)]
        self._learning_rate = piecewise_constant(self.num_sweeps_visited,
                                                 boundaries, values)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grad_var_pairs = zip(self.scaled_grads, self.trainables)
        self._optimize_op = optimizer.apply_gradients(
            grad_var_pairs, global_step=self.num_sweeps_visited)

    @property
    def num_sweeps_visited(self):
        """ A 0-D int32 Tensor. """
        return self._num_sweeps_visited

    @property
    def learning_rate(self):
        """ A 0-D float32 Tensor. """
        return self._learning_rate

    @property
    def trainables(self):
        """ A list of trainable variables that will be updated. """
        return self._trainables

    @property
    def raw_grads(self):
        """ A list of raw gradient Tensors. """
        return self._raw_grads

    @property
    def scaled_grads(self):
        """ A list of scaled gradient Tensors. """
        return self._scaled_grads

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op
