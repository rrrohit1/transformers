# transformers
Playing around with attention and transformers

# Transformer models
Context: a vector (an array of numbers, basically) in the case of machine translation
The encoder and decoder tend to both be recurrent neural networks
the size of the context vector: the number of hidden units in the encoder RNN.
The last hidden state in the encoding process is the context that we pass to the decoder.
Limitation: challenging to deal with longer sentecnes

Attention: allows the model to focus on the relevant parts of the input sequence
Encoder passes all the hidden states, instead of passing the last hidden state
Each hidden state associateed with a word in input sentence and is given a score by which it is multiplied to amplify and depresss hidden state with lower scores.
Attention & Decoder: TBD


# References
1. https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/