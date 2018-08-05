"""
Sample from a character-level language model.
"""

import argparse
import sys

from anyrl.utils.tf_state import load_vars
import numpy as np
import tensorflow as tf
from xformer import LimitedTransformerCell, positional_encoding


def main():
    args = arg_parser().parse_args()

    inputs_ph = tf.placeholder(tf.float32, shape=[1, 256])

    embedded = tf.layers.dense(inputs_ph, args.dimension, name='embed',
                               kernel_initializer=tf.truncated_normal_initializer())
    pos_encoding = positional_encoding(args.context_size, args.dimension)
    cell = LimitedTransformerCell(pos_encoding, num_layers=5)
    state = cell.zero_state(1, tf.float32)
    with tf.variable_scope('rnn'):
        outputs, new_state = cell(embedded, state)
    logits = tf.layers.dense(outputs, 256, name='softmax', activation=None)
    dist = tf.distributions.Categorical(logits=logits)
    raw_samples = dist.sample()
    samples = tf.one_hot(raw_samples, 256, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_vars(sess, args.model, var_list=tf.global_variables(), relaxed=True)
        cur_state = sess.run(state)
        cur_inputs = np.zeros([1, 256], dtype='float32')
        for _ in range(args.context_size):
            feed_dict = {x: y for x, y in zip(state, cur_state)}
            feed_dict[inputs_ph] = cur_inputs
            cur_inputs, cur_raw_samples, cur_state = sess.run([samples, raw_samples, new_state],
                                                              feed_dict=feed_dict)
            sys.stdout.buffer.write(bytes(cur_raw_samples))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model path', default='model.pkl')
    parser.add_argument('--dimension', help='internal model dimension', type=int, default=512)
    parser.add_argument('--context-size', help='bytes per datapoint', type=int, default=128)
    return parser


if __name__ == '__main__':
    main()
