"""
Tests for the Transformer RNNCell.
"""

import numpy as np
import tensorflow as tf

from .transformer import positional_encoding, transformer_layer
from .cell import TransformerCell


def test_basic_equivalence():
    """
    Test that both transformer implementations produce the
    same outputs when applied to a properly-sized
    sequence.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            pos_enc = positional_encoding(4, 6, dtype=tf.float64)
            in_seq = tf.get_variable('in_seq',
                                     shape=(3, 4, 6),
                                     initializer=tf.truncated_normal_initializer(),
                                     dtype=tf.float64)
            cell = TransformerCell(pos_enc, num_heads=2, hidden=24)
            actual, _ = tf.nn.dynamic_rnn(cell, in_seq, dtype=tf.float64)
            with tf.variable_scope('rnn', reuse=True):
                with tf.variable_scope('transformer_cell', reuse=True):
                    expected = transformer_layer(in_seq + pos_enc, num_heads=2, hidden=24)
                    expected = transformer_layer(expected, num_heads=2, hidden=24)
            sess.run(tf.global_variables_initializer())

            actual, expected = sess.run((actual, expected))

            assert not np.isnan(actual).any()
            assert not np.isnan(expected).any()
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected)
