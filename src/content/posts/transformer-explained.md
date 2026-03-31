---
title: "Understanding Transformer: Attention Is All You Need"
published: 2025-11-24
description: A reading reflection on the Transformer architecture -- self-attention, multi-head attention, encoder-decoder structure, and positional encoding.
tags: [Transformer, Attention Mechanism, Deep Learning, NLP]
category: 技术
draft: false
---

## Introduction

Transformer was first proposed in the paper **"Attention is all you need"**. Today, after reading the interpretation of Transformer on [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), I decided to write a reading reflection to analyze the remarkable aspects of Transformer.

Let's start from a high-level perspective. We begin with a specific domain -- machine translation. We input a Chinese sentence and output the corresponding English translation.

From an external perspective without seeing the internal structure of Transformer, we see that our input enters **Encoders**, and then passes through **Decoders** to get the final result.

### Encoder Structure

The encoding component (Encoder) is not just one, but a series of Encoders stacked together. All encoders have the same structure, but their weights are different. Each encoder is divided into two sub-layers:

1. **Self-Attention Layer** - This layer helps the encoder focus on other words in the input sentence while encoding a specific word.
2. **Feed Forward Layer** - A fully connected neural network.

### Input Transformation

For example, a sentence "我 是 一个 学生" (I am a student). We segment it according to our understanding to get these four words.

We use an **"Embedding Algorithm"** to transform each word into a vector.

This embedding only happens in the first encoder layer. The input to subsequent encoders is directly the already encoded vectors. A common principle for all encoders is that they all accept a list of vectors of size 512.

The size of this list is a configurable hyperparameter -- it basically equals the length of the longest sentence in the training dataset.

After we embed the words, we have already prepared two layers of our encoding layer -- **Self Attention** and **Feed Forward layer**.

> **Important Points:**
> - Although X1, X2, X3... appear to enter Self-Attention independently, they are actually related to each other. As each word path flows, they generate associations because "self-attention" means looking at the relationship between this word and other words.
> - When Z1, Z2, Z3... enter Feed-Forward, they are processed independently. Each path can be executed in parallel when flowing through the Feed Forward layer.

## Encoding Process

Each word at each position goes through a self-attention process, and then they each enter a feed-forward neural network -- the same network, but each vector flows through it separately.

## Self-Attention at a High Level

Consider this sentence: *"The animal didn't cross the street because it was too tired"*. What does "it" refer to -- "animal" or "street"?

Actually, we humans can answer this question well, but for machines, this is somewhat difficult to understand. This is where self-attention comes into play.

Transformer's self-attention mechanism allows the model to look at other positions in the input sequence while processing each word, searching for clues that help encode that word.

## Self-Attention in Detail

The first step in calculating self-attention is to create three vectors from each input vector of the encoder (note: already embedded). Therefore, for each word, we create:

- A **Query vector Q**
- A **Key vector K**
- A **Value vector V**

These vectors are obtained by multiplying the word by our trained matrices during training.

**Note:** Here we're talking about single-head attention. Later we'll discuss multi-head attention, which is essentially multiple attentions with weighted summation.

The Query vector Q, Key vector K, and Value vector V are typically 64-dimensional, which is much smaller than the embedding vector of 512. This is an architectural choice because we want to keep the computation of multi-head attention constant.

### Calculating Attention

Suppose we want to calculate the self-attention score for a word "Thinking". The score calculation method is to compute the dot product of the query vector with the key vector of the word to be scored.

The third step is to divide the score value by 8 (the square root of 64, the dimension of the key vector). This yields more stable gradients.

The fourth step is to feed the result into a softmax operation. The softmax operation normalizes the scores so that they are all positive and sum to one.

> **Why do we do this?** Softmax scores determine how often each word appears at that position. Obviously, the word at that position should have the highest softmax score.

The fifth step is to multiply each value vector by the softmax score, preparing to sum them. The purpose is to keep the values of words we want to focus on unchanged and eliminate the influence of irrelevant words.

The sixth step is to sum the weighted value vectors, which produces the output of the self-attention layer at that position. The z1 we get will represent that this word should contain information from this word itself and other words.

## Matrix Calculation on Self-Attention

Each row of X represents a word, then multiplied by their same weight matrices $W^Q$, $W^K$, $W^V$ for a certain head.

Finally, we use our softmax formula to compute the attention output.

## The Beast With Many Heads

We now add the "multi-head" attention mechanism to further improve the self-attention layer.

Previously, we had one Q, K, V. Now we generate multiple QKV and their corresponding training matrices. For example, if we have eight heads, we'll get eight Z matrices through softmax.

These 8 Z matrices will finally get a final Z matrix according to a weight matrix.

The complete process: starting from the input sentence, each word is first converted into a vector through embedding. The model projects this matrix X into different Query, Key, and Value representations, and does this for "8 heads" simultaneously.

By multiplying X with these weight matrices respectively, the model generates three new representations for each head. The reason for having multiple heads is that Transformer hopes to understand the structure of the same sentence from multiple different "angles" or "attention directions". Each head will be smaller (e.g., dimension reduced from 512 to 64), but because there are 8 heads, they together cover "multiple possible attention directions".

After all eight heads complete the attention operation, the model concatenates these output matrices column by column to form a larger matrix Z. A large weight matrix $W^O$ then mixes the multi-head results so that information between different heads can interact or reorganize.

## Positional Encoding

So far, the model we described still lacks a way to consider the order of words in the input sequence. Transformer adds a vector to each input embedding. These vectors follow a specific pattern learned by the model, which helps the model determine the position of each word, or the distance between different words in the sequence.

## Residual Connections

Each encoder has a residual connection around each sub-layer (self-attention, FFNN), and then a layer normalization step is performed.

## Decoder

The encoder processes the input sequence. The output of the top encoder is transformed into a set of attention vectors K and V, which each decoder uses in its "encoder-decoder attention".

### The Fundamental Role of Decoder

> The decoder is responsible for predicting the next word based on the already generated output (e.g., the partially translated sentence) and combined with the full sentence understanding provided by the encoder.

In simpler terms:

- **Encoder** at the input end: Understands the meaning of the entire sentence
- **Decoder** at the output end: Decides the next word to generate based on "what words have I already generated" and "the meaning of the source sentence"
- The whole process is executed repeatedly: one word at a time until the complete output is generated

Therefore, the decoder is **the core of the language generation process**.

### Three Core Functions of Decoder

#### Function 1: Understanding the Generated Partial Output (Self-Attention + Masking)

When the decoder generates a sentence, it doesn't produce it all at once, but word by word. For example, when translating "Je suis heureux", if the decoder has already generated "I am", it now needs to predict "happy".

To do this, it needs to understand the words already generated and their relationships, but cannot "peek" at future words -- because that would violate the autoregressive generation principle.

Therefore, **Mask (future positions are masked)** is added to the decoder's self-attention.

#### Function 2: Using Encoder's Context Information (Cross-Attention)

The decoder needs to use the full sentence understanding provided by the encoder. This is called **Encoder-Decoder Cross Attention**.

The role is: to let the decoder locate the correct information based on the overall meaning of the source sentence when generating each word.

For example, when translating "bank", is it "银行" (bank) or "河岸" (riverbank)? We must combine the complete context of the input, which is the capability of Cross-Attention.

#### Function 3: Converting Semantic Information into Specific Words (Feed-Forward + Linear + Softmax)

When the decoder understands what it has generated through self-attention, and understands the source sentence through cross-attention, it combines this information, then:

1. Enter FFN (two-layer feedforward network)
2. Go through linear layer + Softmax
3. Get the probability distribution of the next word

Finally, select the most likely next word.

## Summary

**Key Takeaways:**

- Transformer revolutionized NLP by using attention mechanisms instead of recurrence or convolution
- Self-attention allows each word to attend to all other words in the sequence
- Multi-head attention enables the model to focus on different types of information simultaneously
- The encoder-decoder architecture with cross-attention enables effective sequence-to-sequence tasks
- Positional encoding and residual connections are crucial for model performance

## Learning Outcomes

Through this reading and reflection, I have gained:

- Deep understanding of Transformer architecture and its components
- Comprehension of self-attention mechanism and its calculation process
- Understanding of multi-head attention and why it's powerful
- Knowledge of encoder-decoder architecture and their roles
- Insight into how Transformer processes sequential data without recurrence
- Appreciation for the elegance of the "attention is all you need" design philosophy

> **Reflection:** The Transformer architecture demonstrates that complex problems can be solved with elegant solutions. By focusing on attention mechanisms, the model achieves state-of-the-art performance across various NLP tasks, proving that sometimes "attention is all you need."
