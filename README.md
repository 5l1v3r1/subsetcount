# subsetcount

Here's a question: what fraction of sentences don't use the letters "a", "b", or "c"? In other words, if you randomly sample a sentence, what is the probability that it won't use the letters "a", "b", or "c"? This might seem difficult to answer accurately. One way to answer the question would be to look at all the books in the world, and simply *count* the number of sentences that meet the criteria. However, if the probability is very low, this sample estimate might easily give us a flat 0.

It's possible to use a language model to get a more accurate answer to the above question. With a language model, we can explicitly sample a sequence using a subset of characters (masking out the disallowed characters), and multiply the estimated probability by `(1-P(forbidden characters))` at each timestep. This gives an unbiased sample estimate of the probability that a sampled sequence will not use a disallowed character.

# Results

I trained a character-level Transformer on the [Amazon Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) corpus for a few days on a single GPU. The resulting per-character loss was 0.84 nats, which is not very good. However, the model still generates fully formed words and often produces coherent sentences.

Using the above language model, I measured some probabilities. All of these are probabilities that the model will produce 256-character chunks without using some subset of characters:

| Characters omitted | Log probability (base e) |
|--------------------|--------------------------|
| aA                 | -18.11                   |
| jqxzJQXZ           | -1.23                    |
| abcABC             | -28.47                   |