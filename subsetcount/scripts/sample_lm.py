"""
Sample from a character-level language model.
"""

import argparse
import sys

from anyrl.utils.tf_state import load_vars
import numpy as np
import tensorflow as tf

from subsetcount.transformer import positional_encoding, transformer_layer


def main():
    args = arg_parser().parse_args()

    sequence = tf.placeholder(tf.float32, shape=(args.context_size, 256))
    cur_sequence = np.zeros((args.context_size, 256), dtype='float32')

    outputs = tf.concat([tf.zeros_like(sequence[:, :1]), sequence[:, :-1]], axis=1)
    outputs = tf.layers.dense(outputs, args.dimension, name='embed',
                              kernel_initializer=tf.truncated_normal_initializer())
    outputs += positional_encoding(args.context_size, args.dimension)
    outputs = tf.expand_dims(outputs, axis=0)
    with tf.variable_scope('model'):
        for _ in range(5):
            outputs = transformer_layer(outputs)
    outputs = outputs[0]
    logits = tf.layers.dense(outputs, 256, name='softmax', activation=None)
    dist = tf.distributions.Categorical(logits=logits)
    raw_samples = dist.sample()
    samples = tf.one_hot(raw_samples, 256, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_vars(sess, args.model, var_list=tf.global_variables(), relaxed=True)
        for i in range(args.context_size):
            cur_samples, cur_raw_samples = sess.run([samples, raw_samples],
                                                    feed_dict={sequence: cur_sequence})
            cur_sequence[i] = cur_samples[i]
            sys.stderr.buffer.write(bytes([cur_raw_samples[i]]))
            sys.stderr.buffer.flush()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model path', default='model.pkl')
    parser.add_argument('--dimension', help='internal model dimension', type=int, default=512)
    parser.add_argument('--context-size', help='bytes per datapoint', type=int, default=128)
    return parser


if __name__ == '__main__':
    main()
