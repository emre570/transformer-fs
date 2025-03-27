import torch
import torch.nn as nn
from mh_attention import MultiHeadAttention
from feed_forward import FeedForwardBlock
from decoder import DecoderBlock, Decoder

# Dummy shapes
batch_size = 8
seq_len_src = 4
seq_len_tgt = 4
d_model = 8
heads = 2
num_layers = 3

# Random tensors
x = torch.rand(batch_size, seq_len_tgt, d_model)  # Decoder input
encoder_output = torch.rand(batch_size, seq_len_src, d_model)  # Encoder output

# src_mask: no padding for now
src_mask = torch.ones((batch_size, 1, 1, seq_len_src)).bool()
print(f"src_mask shape: {src_mask.shape}")

# tgt_mask: look-ahead mask
tgt_mask = torch.tril(torch.ones((seq_len_tgt, seq_len_tgt))).bool()
tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
tgt_mask = tgt_mask.expand(batch_size, 1, seq_len_tgt, seq_len_tgt)  # (batch, 1, tgt_len, tgt_len)
print(f"tgt_mask shape: {tgt_mask.shape}")

# Modules
decoder_blocks = nn.ModuleList([
    DecoderBlock(
        MultiHeadAttention(d_model, heads, dropout=0.1),
        MultiHeadAttention(d_model, heads, dropout=0.1),
        FeedForwardBlock(d_model, d_model * 4, dropout=0.1),
        dropout=0.1
    )
    for _ in range(num_layers)
])

# Decoder
decoder = Decoder(layers=decoder_blocks)

# Forward pass
out = decoder(x, encoder_output, src_mask, tgt_mask)
print("Stacked Decoder output shape:", out.shape)
