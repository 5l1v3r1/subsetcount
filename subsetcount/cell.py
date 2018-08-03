"""
Using Transformers as RNN cells.

This helps for two things:

  1. Faster sampling.
  2. Compatibility with RNN-based code.

"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell  # pylint: disable=E0611

from .transformer import split_heads


class TransformerCell(RNNCell):
    """
    An RNNCell that implements a Transformer.

    Back-propagation through this RNN is not efficient,
    since it is not properly batched.
    However, the forward pass is as fast as it could be.

    State tuples are of the form:

        (current_step, keys_1, values_1, keys_2, values_2, ...)

    """

    def __init__(self,
                 pos_encoding,
                 num_layers=6,
                 num_heads=8,
                 hidden=2048,
                 trainable=True,
                 name=None,
                 dtype=None):
        """
        Create a new Transformer cell.

        Args:
          pos_encoding: a positional encoding Tensor. The
            time and inner dimensions must both be known
            statically, since the state shape depends on
            both things.
          num_layers: the number of layers.
          num_heads: the number of attention heads.
          hidden: the FC hidden layer size.
          trainable: use trainable variables.
          name: the scope name.
          dtype: the datatype.
        """
        super(TransformerCell, self).__init__(trainable=trainable, name=name, dtype=dtype)
        self.pos_encoding = pos_encoding
        self.num_layers = 6
        self.num_heads = num_heads
        self.hidden = hidden

    @property
    def time_horizon(self):
        """
        Get the number of timesteps that the model can see
        at once.
        """
        return self.pos_encoding.get_shape()[0].value

    @property
    def state_size(self):
        return [tf.TensorShape(())] + [self.pos_encoding.get_shape()] * 2 * self.num_layers

    @property
    def output_size(self):
        return self.pos_encoding.get_shape()[1].value

    def zero_state(self, batch_size, dtype):
        zero_pos = tf.zeros(batch_size, dtype=dtype)
        zeros = tf.zeros([batch_size, self.time_horizon, self.output_size], dtype=dtype)
        return (zero_pos,) + (zeros,) * (self.num_layers * 2)

    def call(self, inputs, state):  # pylint: disable=W0221
        layer_inputs = inputs
        new_states = [tf.clip_by_value(state[0] + 1.0, 0, self.time_horizon - 1)]
        timestep_idxs = tf.cast(state[0], tf.int32)
        # TODO: add positional encoding.
        for layer_idx in range(self.num_layers):
            keys = state[1 + layer_idx * 2]
            values = state[2 + layer_idx * 2]
            new_keys, new_values, layer_inputs = self.attention_layer(
                timestep_idxs,
                keys,
                values,
                layer_inputs,
            )
            layer_inputs = self.fc_layer(layer_inputs)
            new_states.extend([new_keys, new_values])
        return layer_inputs

    def attention_layer(self, timestep_idxs, keys, values, inputs, scope='attention'):
        """
        Apply masked attention for a single timestep.

        Args:
          timestep_idxs: a 1-D Tensor of indices.
          keys: the key history for the layer.
          values: the value history for the layer.
          inputs: the inputs for the current timesteps.
          scope: the scope name.

        Returns:
          A tuple (outputs, new_keys, new_values):
            outputs: a [batch x N] Tensor from the layer.
            new_keys: the new key history.
            new_values: the new value history.
        """
        with tf.variable_scope(None, default_name=scope):
            projected = tf.layers.dense(inputs, self.output_size * 3,
                                        name='key_query_value')
            projected = tf.expand_dims(projected, axis=1)

            # Resulting shape: [batch x 1 x N]
            next_keys, next_queries, next_values = tf.split(projected, 3, axis=-1)

            keys = inject_at_timestep(timestep_idxs, keys, next_keys[0, :])
            values = inject_at_timestep(timestep_idxs, values, next_values[0, :])

            # Resulting shape: [batch x heads x timesteps x N/heads]
            split_keys = split_heads(keys, self.num_heads)
            split_values = split_heads(values, self.num_heads)

            # Resulting shape: [batch x heads x N/heads]
            split_queries = split_heads(next_queries, self.num_heads)[:, :, 0]

            # TODO: apply attention on top of new keys/values with queries.

            return keys, values, outputs

    def fc_layer(self, inputs):
        """
        Apply the fully-connected layer.

        Args:
          inputs: a [batch x N] input.

        Returns:
          A [batch x N] output.
        """
        # TODO: this.
        pass


def inject_at_timestep(timestep_idxs, sequence, new_values):
    """
    Inject a batch of values at respective timesteps of a
    sequence.

    Args:
      sequence: a [batch x timesteps x N] sequence.
      timestep_idxs: a 1-D Tensor of indices.
      new_values: a [batch x N] Tensor where each batch
        element should be injected into the given timestep
        index of the input sequence.

    Returns:
      A [batch x timestep x N] sequence.
    """
    # Create a [batch x timestep] Tensor of timesteps.
    ranges = tf.range(tf.shape(sequence)[0], dtype=timestep_idxs.dtype)
    ranges = tf.tile(tf.expand_dims(ranges, axis=0), [tf.shape(sequence)[0], 1])

    # Create a mask of shape [batch x timestep x N].
    mask = tf.equal(ranges - tf.expand_dims(timestep_idxs, axis=-1), tf.zeros_like(ranges))
    mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, tf.get_shape(sequence)[-1].value])

    new_seq = tf.zeros_like(sequence) + tf.expand_dims(new_values, axis=1)
    return tf.where(mask, new_seq, sequence)
