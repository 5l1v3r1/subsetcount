"""
Approximate the fraction of samples from a language model
that do not use a set of characters.
"""

from anyrl.utils.tf_state import load_vars
import numpy as np
import tensorflow as tf
from xformer import LimitedTransformerCell, positional_encoding

from subsetcount.sample_lm import arg_parser


def main():
    parser = arg_parser()
    parser.add_argument('--omit-chars', help='characters to omit', default='jqxzJQXZ')
    args = parser.parse_args()

    mask = [True] * 256
    for ch in args.omit_chars:
        mask[ord(ch)] = False
    log_probs = subset_probs(args.dimension, args.context_size, mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_vars(sess, args.model, var_list=tf.global_variables(), relaxed=True)
        log_sum = -np.inf
        count = 0.0
        while True:
            probs = sess.run(log_probs)
            for prob in probs:
                count += 1.0
                log_sum = np.logaddexp(log_sum, prob)
            print('count %d: log_prob=%f' % (int(count), log_sum - np.log(count)))


def subset_probs(dimension, context_size, character_mask, batch_size=32):
    """
    Sample sequences using a subset of characters and get
    importance sampling probabilities for them.

    Args:
      dimension: the model dimension.
      context_size: the sequence length.
      character_mask: a boolean array indicating which
        bytes to sample and which to avoid. This is a 1-D
        vector where True means a character may be used.

    Returns:
      A vector of log probabilities, one for each sample,
        where the probability is a sample estimate for the
        probability that all sampled characters are
        allowed.
    """
    mask = tf.constant(np.array([character_mask] * batch_size), tf.bool)

    pos_encoding = positional_encoding(context_size, dimension)
    cell = LimitedTransformerCell(pos_encoding, num_layers=5)

    def loop_body(timestep, log_probs, inputs, states):
        embedded = tf.layers.dense(inputs, dimension, name='embed',
                                   kernel_initializer=tf.truncated_normal_initializer())
        with tf.variable_scope('rnn'):
            outputs, new_states = cell(embedded, states)
        logits = tf.layers.dense(outputs, 256, name='softmax', activation=None)

        all_probs = tf.nn.log_softmax(logits)
        neg_inf = tf.negative(tf.zeros_like(logits) * np.inf)
        allowed_probs = tf.where(mask, all_probs, neg_inf)
        log_probs += tf.reduce_sum(allowed_probs, axis=-1)

        masked_logits = tf.where(mask, logits, neg_inf)
        dist = tf.distributions.Categorical(logits=masked_logits)
        samples = tf.one_hot(dist.sample(), 256, dtype=tf.float32)

        return timestep + 1, log_probs, samples, new_states

    counter = tf.zeros((), dtype=tf.int32)
    log_probs = tf.zeros([batch_size], dtype=tf.float32)
    inputs = tf.zeros([batch_size, 256], dtype=tf.float32)
    states = cell.zero_state(batch_size, tf.float32)

    return tf.while_loop(cond=lambda timestep, _x, _y, _z: timestep < context_size,
                         body=loop_body,
                         loop_vars=(counter, log_probs, inputs, states))[1]


if __name__ == '__main__':
    main()
