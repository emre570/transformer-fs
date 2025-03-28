# 🧠 Transformer Cheat Sheet (Architecture & Parameters)

---

## 🔁 Visual Flow (From Input to Output)

```text
1. Input sentence (src_ids): [17, 42, 31]
   ↓
2. Embedding → src_embed(src_ids)
   → shape: (batch, seq_len_src, d_model)

3. Add positional encoding → src_pos
   → still shape: (batch, seq_len_src, d_model)

4. Encoder (stack of N blocks):
   → Self-attention + FFN
   → Output: encoder memory (context)

======== Switch to Decoder ========

5. Decoder input (tgt_ids): [8, 23, 77]
   ↓
6. Embedding → tgt_embed(tgt_ids)
   ↓
7. Add positional encoding → tgt_pos
   ↓
8. Decoder (stack of N blocks):
   → Masked self-attn
   → Cross-attn (queries from decoder, keys/values from encoder)
   → FFN
   → Output: (batch, seq_len_tgt, d_model)

9. Projection layer → Linear(d_model, tgt_vocab_size)
   ↓
10. Log-softmax → probability over target vocab
```

---

## 🧩 Transformer Component Map

| **Component**        | **What It Does** |
|----------------------|------------------|
| `src_embed`          | Embeds source tokens (IDs → vectors) |
| `tgt_embed`          | Embeds target tokens |
| `src_pos`            | Adds positional info to source embeddings |
| `tgt_pos`            | Adds positional info to target embeddings |
| `Encoder`            | Stack of `N` blocks: self-attn + FFN |
| `Decoder`            | Stack of `N` blocks: self-attn + cross-attn + FFN |
| `MultiHeadAttention` | Attention with `Q`, `K`, `V` split into `heads` |
| `FeedForwardBlock`   | 2-layer MLP (d_model → d_ff → d_model) |
| `ResidualConnection` | Add & LayerNorm wrapper |
| `ProjectionLayer`    | Linear(d_model → tgt_vocab_size) + log softmax |
| `src_mask`           | Ignores `<PAD>` tokens in encoder input |
| `tgt_mask`           | Prevents decoder from seeing future tokens |

---

## 🔧 Core Hyperparameters

| **Name**       | **Meaning** |
|----------------|-------------|
| `d_model`      | Dimensionality of embeddings (common: 512) |
| `heads`        | Number of attention heads (common: 8) |
| `N`            | Number of encoder/decoder layers (common: 6) |
| `d_ff`         | Feedforward hidden size (usually 4× d_model) |
| `src_vocab_size` | Size of source language vocab |
| `tgt_vocab_size` | Size of target language vocab |
| `dropout`      | Regularization (e.g. 0.1) |

---

## 🧪 Output Shape Cheat Sheet

| **Layer**              | **Output Shape** |
|------------------------|------------------|
| `Embedding`            | `(batch, seq_len, d_model)` |
| `Encoder`              | `(batch, seq_len_src, d_model)` |
| `Decoder`              | `(batch, seq_len_tgt, d_model)` |
| `ProjectionLayer`      | `(batch, seq_len_tgt, tgt_vocab_size)` |
| `Logits` after softmax | `(batch, seq_len_tgt, tgt_vocab_size)` |
