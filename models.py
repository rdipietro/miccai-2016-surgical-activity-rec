from __future__ import division, print_function

import numpy as np
import tensorflow as tf

import data


class LSTM(object):

    def __init__(self, inputs, resets, training, num_layers, hidden_layer_size,
                 init_scale, dropout_keep_prob):
        """ Create a long short-term memory RNN.

        This maps RNN inputs to RNN outputs. Computing predictions from the RNN
        outputs needs to happen elsewhere.

        Args:
            inputs: A 3-D float32 Tensor with shape
                `[batch_size, duration, input_size]`.
            resets: A 3-D bool Tensor with shape
                `[batch_size, duration, 1]`. These indicate when sequences
                reset (to handle states appropriately with sequences that have
                been wrapped to form a sweep).
            training: A 0-D bool Tensor. When False, dropout won't be applied.
            num_layers: An integer. The number of hidden layers.
            hidden_layer_size: An integer. The number of hidden units per layer.
            init_scale: A float. All weights will be initialized using a
                uniform distribution over `[-init_scale, init_scale]`.
            dropout_keep_prob: A float. The fraction of inputs to keep whenever
                dropout is applied. Dropout is applied to the inputs/outputs
                of each time step, and is never applied across time steps.
        """

        self.inputs = inputs
        self.resets = resets
        self.training = training
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.init_scale = init_scale
        self.dropout_keep_prob = dropout_keep_prob
        self.input_size = inputs.get_shape().as_list()[2]

        batch_states_shape = tf.pack([tf.shape(inputs)[0],
                                      2*self.hidden_layer_size])
        self._batch_start_states = tf.zeros(batch_states_shape,
                                            dtype=tf.float32)
        keep_prob = tf.cond(training,
                            lambda: tf.constant(self.dropout_keep_prob),
                            lambda: tf.constant(1.0))

        initializer = tf.random_uniform_initializer(-self.init_scale,
                                                    self.init_scale)
        with tf.variable_scope('LSTM', initializer=initializer):

            # We need to swap the batch, time axes since tf.scan will split
            # along dimension 0.
            inputs = tf.transpose(inputs, [1, 0, 2])
            resets = tf.transpose(resets, [1, 0, 2])

            states_list = []
            prev_layer_outputs = tf.nn.dropout(inputs, keep_prob)
            for layer in xrange(self.num_layers):

                def fixed_size_lstm_block(c_prev_and_m_prev, x_and_r):
                    if layer == 0:
                        block_input_size = self.input_size
                    else:
                        block_input_size = self.hidden_layer_size
                    return self._lstm_block(c_prev_and_m_prev, x_and_r,
                                            block_input_size)

                x_and_r = tf.concat(2, [prev_layer_outputs,
                                        tf.cast(resets, tf.float32)])
                with tf.variable_scope('layer%d' % layer):
                    c_and_m = tf.scan(fixed_size_lstm_block, x_and_r,
                                      initializer=self._batch_start_states)
                states_list.append(c_and_m)
                prev_layer_outputs = tf.nn.dropout(
                    c_and_m[:, :, self.hidden_layer_size:], keep_prob)

            _states = tf.concat(2, [tf.expand_dims(states, 2)
                                    for states in states_list])

            # Now put the batch, time axes back.
            self._states = tf.transpose(_states, [1, 0, 2, 3])
            self._outputs = tf.transpose(prev_layer_outputs, [1, 0, 2])

    def _lstm_block(self, c_prev_and_m_prev, x_and_r, block_input_size):
        """ LSTM block.

        This implementation uses a forget gate and peephole connections. Also,
        sequence resets `r` are used here to handle state resets internally
        instead of externally.

        Args:
            c_prev_and_m_prev: A 2-D float32 Tensor with shape
                `[batch_size, 2*hidden_layer_size]`.
            x_and_r: A 2-D float32 Tensor with shape
                `[batch_size, block_input_size+1]`. It's a concatenation of
                the block's inputs with sequence-reset booleans (though with
                type float32). In the case of multiple layers, the inputs are
                the previous layer's outputs.
            block_input_size: An integer.

        Returns:
            The updated state c_and_m, with the same shape as
            `c_prev_and_m_prev`.
        """

        def xmul(tensor, weights_name):
            W = tf.get_variable(weights_name, shape=[block_input_size,
                                                     self.hidden_layer_size])
            return tf.matmul(tensor, W)

        def mmul(tensor, weights_name):
            W = tf.get_variable(weights_name, shape=[self.hidden_layer_size,
                                                     self.hidden_layer_size])
            return tf.matmul(tensor, W)

        def diagcmul(tensor, weights_name):
            w = tf.get_variable(weights_name, shape=[self.hidden_layer_size])
            return tensor*w

        def bias(name):
            b = tf.get_variable(name, shape=[self.hidden_layer_size],
                                initializer=tf.constant_initializer(0.0))
            return b

        x = x_and_r[:, :block_input_size]
        r = tf.cast(x_and_r[:, block_input_size], tf.bool)

        # If r[i] is True, revert back to initial states. Otherwise, keep
        # the states from the previous time step.
        c_prev_and_m_prev = tf.select(r,
                                      self._batch_start_states,
                                      c_prev_and_m_prev)
        c_prev, m_prev = tf.split(1, 2, c_prev_and_m_prev)

        x_tilde = tf.tanh( xmul(x, 'W_xx') +
                           mmul(m_prev, 'W_xm') + bias('b_x') )
        i = tf.sigmoid( xmul(x, 'W_ix') + mmul(m_prev, 'W_im') +
                        diagcmul(c_prev, 'w_ic') + bias('b_i') )
        f = tf.sigmoid( xmul(x, 'W_fx') + mmul(m_prev, 'W_fm') +
                        diagcmul(c_prev, 'w_fc') + bias('b_f') )
        c = x_tilde*i + c_prev*f
        o = tf.sigmoid( xmul(x, 'W_ox') + mmul(m_prev, 'W_om') +
                        diagcmul(c, 'w_oc') + bias('b_o') )
        m = o*tf.tanh(c)

        c_and_m = tf.concat(1, [c, m])
        return c_and_m

    @property
    def states(self):
        """ A 4-D float32 Tensor with shape
        `[batch_size, duration, num_layers, hidden_layer_size]`. """
        return self._states

    @property
    def outputs(self):
        """ A 3-D float32 Tensor with shape
        `[batch_size, duration, hidden_layer_size]`. """
        return self._outputs


class LSTMModel(object):

    def __init__(self, input_size, target_size, num_layers, hidden_layer_size,
                 init_scale, dropout_keep_prob):
        """ A base class for LSTM models that includes predictions and loss.

        Args:
            input_size: An integer. The number of inputs per time step.
            target_size: An integer. The dimensionality of the one-hot
                encoded targets.
            num_layers: An integer. The number of hidden layers.
            hidden_layer_size: An integer. The number of hidden units per layer.
            init_scale: A float. All weights will be initialized using a
                uniform distribution over `[-init_scale, init_scale]`.
            dropout_keep_prob: A float. The fraction of inputs to keep whenever
                dropout is applied. Dropout is applied to the inputs/outputs
                of each time step, and is never applied across time steps.
        """

        self.input_size = input_size
        self.target_size = target_size
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.init_scale = init_scale
        self.dropout_keep_prob = dropout_keep_prob

        self._inputs = tf.placeholder(
            tf.float32, shape=[None, None, input_size], name='inputs')
        self._resets = tf.placeholder(
            tf.bool, shape=[None, None, 1], name='resets')
        self._targets = tf.placeholder(
            tf.float32, shape=[None, None, target_size], name='targets')
        self._training = tf.placeholder(tf.bool, shape=[], name='training')

        outputs = self._compute_rnn_outputs()
        output_size = self._compute_rnn_output_size()

        initializer = tf.random_uniform_initializer(-self.init_scale,
                                                    self.init_scale)
        with tf.variable_scope('logits', initializer=initializer):
            W = tf.get_variable('W', shape=[output_size, self.target_size])
            b = tf.get_variable('b', shape=[self.target_size])
            outputs_matrix = tf.reshape(outputs, [-1, output_size])
            logits = tf.nn.xw_plus_b(outputs_matrix, W, b)
            batch_size, duration, _ = tf.unpack(tf.shape(self.inputs))
            logits_shape = tf.pack([batch_size, duration, self.target_size])
            self._logits = tf.reshape(logits, logits_shape, name='logits')

        with tf.variable_scope('loss'):
            logits = tf.reshape(self.logits, [-1, self.target_size])
            targets = tf.reshape(self.targets, [-1, self.target_size])
            cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                      targets)
            self._loss = tf.reduce_mean(cross_entropies, name='loss')

    def _compute_rnn_outputs(self):
        """ Compute RNN outputs.

        Returns:
            A 3-D float32 Tensor with shape
            `[batch_size, duration, output_size]`.
        """

        raise NotImplementedError()

    def _compute_rnn_output_size(self):
        """ Compute RNN output size.

        It's not always the same as `hidden_layer_size`: in the
        BidirectionalLSTM case, it's `2*hidden_layer_size`.

        Returns:
            An integer.
        """

        raise NotImplementedError()

    @property
    def inputs(self):
        """ A 3-D float32 Placeholder with shape
        `[batch_size, duration, input_size]`. """
        return self._inputs

    @property
    def resets(self):
        """ A 3-D bool Placeholder with shape `[batch_size, duration, 1]`. """
        return self._resets

    @property
    def targets(self):
        """ A 3-D float32 Placeholder with shape
        `[batch_size, duration, target_size]`. """
        return self._targets

    @property
    def training(self):
        """ A 0-D bool Placeholder. """
        return self._training

    @property
    def logits(self):
        """ A 3-D float32 Tensor with shape
        `[batch_size, duration, target_size]`. """
        return self._logits

    @property
    def loss(self):
        """ A 0-D float32 Tensor. """
        return self._loss


class ForwardLSTMModel(LSTMModel):

    def __init__(self, *args):
        """ Create a forward LSTM model.

        Args:
            See `LSTMModel`.
        """
        super(ForwardLSTMModel, self).__init__(*args)

    def _compute_rnn_outputs(self):
        self._fw_lstm = LSTM(self.inputs, self.resets, self.training,
                             self.num_layers, self.hidden_layer_size,
                             self.init_scale, self.dropout_keep_prob)
        return self._fw_lstm.outputs

    def _compute_rnn_output_size(self):
        return self._fw_lstm.hidden_layer_size


class ReverseLSTMModel(LSTMModel):

    def __init__(self, *args):
        """ Create a reverse LSTM model.

        Args:
            See `LSTMModel`.
        """
        super(ReverseLSTMModel, self).__init__(*args)

    def _compute_rnn_outputs(self):
        reversed_inputs = tf.reverse(self.inputs, [False, True, False])
        reversed_resets = tf.reverse(self.resets, [False, True, False])
        self._rv_lstm = LSTM(reversed_inputs, reversed_resets, self.training,
                             self.num_layers, self.hidden_layer_size,
                             self.init_scale, self.dropout_keep_prob)
        outputs = tf.reverse(self._rv_lstm.outputs, [False, True, False])
        return outputs

    def _compute_rnn_output_size(self):
        return self._rv_lstm.hidden_layer_size


class BidirectionalLSTMModel(LSTMModel):

    def __init__(self, *args):
        """ Create a bidirectional LSTM model.

        Args:
            See `LSTMModel`.
        """
        super(BidirectionalLSTMModel, self).__init__(*args)

    def _compute_rnn_outputs(self):

        reversed_inputs = tf.reverse(self.inputs, [False, True, False])
        reversed_resets = tf.reverse(self.resets, [False, True, False])
        with tf.variable_scope('fw'):
            self._fw_lstm = LSTM(self.inputs, self.resets, self.training,
                                 self.num_layers, self.hidden_layer_size,
                                 self.init_scale, self.dropout_keep_prob)
        with tf.variable_scope('rv'):
            self._rv_lstm = LSTM(reversed_inputs, reversed_resets,
                                 self.training, self.num_layers,
                                 self.hidden_layer_size, self.init_scale,
                                 self.dropout_keep_prob)

        fw_outputs = self._fw_lstm.outputs
        rv_outputs = tf.reverse(self._rv_lstm.outputs, [False, True, False])
        outputs = tf.concat(2, [fw_outputs, rv_outputs])
        return outputs

    def _compute_rnn_output_size(self):
        return self._fw_lstm.hidden_layer_size + self._rv_lstm.hidden_layer_size


def predict(sess, model, input_seqs, reset_seqs):
    """ Compute prediction sequences from input sequences.

    Args:
        sess: A Session.
        model: An LSTMModel.
        input_seqs: A list of input sequences, each a float32 NumPy array with
            shape `[duration, input_size]`.
        reset_seqs: A list of reset sequences, each a bool NumPy array with
            shape `[duration, 1]`.

    Returns:
        A list of prediction sequences, each a NumPy array with shape
        `[duration, 1]`, containing predicted labels for each time step.
    """

    batch_size = len(input_seqs)
    seq_durations = [len(seq) for seq in input_seqs]
    input_sweep, reset_sweep = data.sweep_generator(
        [input_seqs, reset_seqs], batch_size=batch_size).next()

    logit_sweep = sess.run(model.logits, feed_dict={model.inputs: input_sweep,
                                                    model.resets: reset_sweep,
                                                    model.training: False})

    logit_seqs = [seq[:duration]
                  for (seq, duration) in zip(logit_sweep, seq_durations)]
    prediction_seqs = [np.argmax(seq, axis=1).reshape(-1, 1)
                       for seq in logit_seqs]

    return prediction_seqs
