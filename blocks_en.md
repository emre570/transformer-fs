# Transformer Architecture: Building Blocks Explained

Hey there! In this post, I’m going to walk you through the building blocks of the Transformer architecture. But don’t worry—this isn’t going to be one of those dry academic reads. Think of it more like we're sitting down for a coffee and chatting about how all of this works. The goal? No more “What the heck is this, bro?” moments. Everything will be clear, with examples and just enough math to make it stick.

This article is based on Umar Jamil's video [Coding a Transformer from scratch on PyTorch](tab:https://www.youtube.com/watch?v=ISNdQcPhsts).

Ready? Let’s dive in.

---

## Input Embeddings

Models can’t understand words directly. If you type "dog", "hello", or "GPT", it’s just gibberish to the model. So, the first step is to convert each word into a numerical vector. The bigger the vector (say 512 dimensions), the more information it can carry about the meaning of the word.

Here’s how we do it in PyTorch:

```python
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

Why scale by √d_model? Because working with tiny numbers slows down learning. This factor keeps the values in a good range, making training more stable.

---

## Positional Encoding

We’ve got our words turned into vectors—but the model still has no idea where in the sentence each word is. “I went home” and “Home I went” would produce the same embeddings. Not good.

To fix this, we add **positional information** to each word’s embedding. We generate a special vector for each position in the sentence and add it to the word vector. The formulas look like this:

$$
\[PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\]
\[PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\]
$$

Why sine and cosine? These functions are periodic, which helps the model learn distances between words (like word 5 and word 10). Using both sin and cos lets us encode directionality, too.

Quick example:

- Sentence: "I am going home"
- Each position gets a unique vector based on sin/cos and gets added to the word embedding.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

---

## Multi-Head Attention

Now we get to the core of it all. Attention is how the model asks: “Is this word related to that word?” Multi-Head Attention lets the model ask that question from **multiple perspectives**.

We use three key components:
- **Query (Q):** The word we're focusing on ("What should I pay attention to?")
- **Key (K):** The identity of other words ("What’s available to look at?")
- **Value (V):** The actual information we pull from those other words

Formula:
$$
\[Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V\]
$$

Example:
> Sentence: "Ayşe threw the ball to Ali because _ was tired."
>
> What goes in the blank? Ayşe or Ali?
>
> The model focuses on the word “because” (Query), compares it to all other words (Keys), calculates similarity scores, then pulls info from the most relevant ones (Values).

Why multiple heads?
Each head captures a different type of relationship—grammar, emotion, timing, etc.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.h = h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        def transform(x, linear):
            x = linear(x)
            x = x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            return x

        q = transform(q, self.w_q)
        k = transform(k, self.w_k)
        v = transform(v, self.w_v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)
```

---

## Feed Forward Network

Now that we’ve modeled relationships between words, it’s time to dig into each word individually and extract more complex features.

Every position goes through the same MLP (two linear layers):

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

- First layer expands the dimension (d_model → d_ff)
- ReLU adds non-linearity
- Second layer brings it back down (d_ff → d_model)

Each word gets a deeper representation—but we return to the original shape so we can keep stacking blocks.

---

## Layer Normalization

Sometimes, activations between layers can get out of control. Values too big or too small make learning hard. That’s where LayerNorm comes in.

It normalizes across each input’s feature dimension:

$$
\[LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\]
$$

Where:
- `x`: input vector
- `μ`: mean
- `σ²`: variance
- `ε`: small constant for numerical stability
- `γ`, `β`: learnable parameters to scale/shift the result

Note: We use LayerNorm instead of BatchNorm because LayerNorm works per example (not per batch), making it more suitable for sequence models.

```python
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

---

## Residual Connection

Deep networks tend to forget what the input was. Residual connections fix that by adding the original input back in:

```python
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

This helps gradients flow more easily and allows the model to go deeper without losing track of the original signal. Every Transformer block uses this trick.

---