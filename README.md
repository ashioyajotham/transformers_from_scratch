# Building Transformers from Scratch
* My second attempt at building transformers from scratch using the [Attention paper](https://arxiv.org/abs/1706.03762) as a guide.
* Special thanks to [Joris Baan](https://github.com/jsbaan/transformer-from-scratch) for the original code and the inspiration to build this project.

## Introduction
* Transformers have become the go-to model for many natural language processing tasks. They have been shown to outperform RNNs and LSTMs on many tasks. The transformer model was introduced in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. The transformer model is based on the self-attention mechanism, which allows the model to focus on different parts of the input sequence when making predictions. The transformer model consists of an encoder and a decoder, each of which is composed of multiple layers of self-attention and feed-forward neural networks. The transformer model has been shown to achieve state-of-the-art performance on many natural language processing tasks, including machine translation, text summarization, and question answering.

* In this project, I will build a transformer model from scratch using PyTorch. The model will be trained on a simple dataset and will be evaluated on a test set. The goal of this project is to gain a better understanding of how the transformer model works and how it can be implemented in code.

* The goal of this project is to build a transformer model from scratch using PyTorch. The model will be trained on a simple dataset and will be evaluated on a test set. The model will be built using the following components:
    - Multi-Head Attention - The model will use multi-head attention to allow the model to focus on different parts of the input sequence when making predictions.
    - Position-wise Feed-Forward Networks - The model will use position-wise feed-forward networks to process the output of the multi-head attention layer.
    - Layer Normalization - The model will use layer normalization to normalize the output of the multi-head attention and feed-forward layers.
    - Residual Connections - The model will use residual connections to allow the model to learn the identity function.
    - Positional Encoding - The model will use positional encoding to encode the position of each token in the input sequence.
    - Masking - The model will use masking to prevent the model from attending to future tokens during training.

* The model will be trained using the Adam optimizer and the learning rate will be scheduled using the Noam learning rate scheduler. The model will be evaluated using the [BLEU score metric](https://en.wikipedia.org/wiki/BLEU).

## The project will be divided into the following sections:
1. Data Preprocessing
2. Model Architecture
3. Training
4. Evaluation
5. Conclusion

* Side note: I was listening to the theory of consciousness from the YouTube video [Consciousness of Artificial Intelligence](https://www.youtube.com/watch?v=sISkAb7suqo) while building this. It's a very interesting video and I highly recommend it.
