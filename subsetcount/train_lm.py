"""
Train a character-level language model.
"""

import argparse
import random

from anyrl.utils.tf_state import load_vars, save_vars
import numpy as np
import tensorflow as tf
from xformer import positional_encoding, transformer_layer


def main():
    args = arg_parser().parse_args()

    dataset = corpus_dataset(args.corpus, args.context_size)
    batch = dataset.batch(args.batch).make_one_shot_iterator().get_next()

    outputs = tf.concat([tf.zeros_like(batch[:, :1]), batch[:, :-1]], axis=1)
    outputs = tf.layers.dense(outputs, args.dimension, name='embed',
                              kernel_initializer=tf.truncated_normal_initializer())
    outputs += positional_encoding(args.context_size, args.dimension)
    with tf.variable_scope('rnn'):
        with tf.variable_scope('transformer'):
            for _ in range(5):
                outputs = transformer_layer(outputs)

    logits = tf.layers.dense(outputs, 256, name='softmax', activation=None)
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch, logits=logits)
    loss = tf.reduce_mean(losses)

    optimize = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
    global_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.int64),
                              name='global_step')
    inc_step = tf.assign_add(global_step, tf.constant(1, dtype=tf.int64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_vars(sess, args.model, var_list=tf.global_variables())
        while True:
            cur_step, cur_loss, _ = sess.run([inc_step, loss, optimize])
            print('step %d: loss=%f' % (cur_step, cur_loss))
            if cur_step % args.save_interval == 0:
                save_vars(sess, args.model, var_list=tf.global_variables())


def corpus_dataset(corpus_path, context_size):
    with open(corpus_path, 'rb') as in_file:
        data = in_file.read()

    def sample_gen():
        while True:
            idx = random.randrange(len(data) - context_size)
            seq = data[idx:idx + context_size]
            one_hots = np.zeros((context_size, 256), dtype='float32')
            one_hots[np.arange(context_size), np.array(list(seq))] = 1.0
            yield one_hots

    return tf.data.Dataset.from_generator(sample_gen, tf.float32, [context_size, 256])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model path', default='model.pkl')
    parser.add_argument('--dimension', help='internal model dimension', type=int, default=512)
    parser.add_argument('--context-size', help='bytes per datapoint', type=int, default=128)

    parser.add_argument('--batch', help='batch size', type=int, default=16)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--save-interval', help='steps per save', type=int, default=10)

    parser.add_argument('corpus', help='A text file where data is gathered from')
    return parser


if __name__ == '__main__':
    main()
