import torch
import torch.nn as nn
from mh_attention import MultiHeadAttention
from feed_forward import FeedForwardBlock
from encoder import EncoderBlock
from layer_norm import LayerNormalization
from residual_connection import ResidualConnection

# Settings
batch_size = 1
seq_len = 4
d_model = 8
heads = 2

x = torch.rand(batch_size, seq_len, d_model)

# Dummy mask: none (assume full valid tokens)
mask = torch.ones((batch_size, 1, 1, seq_len)).bool()

def make_encoder_block():
    return EncoderBlock(
        self_attention_block=MultiHeadAttention(d_model, heads, dropout=0.0),
        feed_forward_block=FeedForwardBlock(d_model, d_model * 4, dropout=0.0),
        dropout=0.0
    )

# Test with different depths
for num_layers in [1, 2, 4, 6]:
    layers = nn.ModuleList([make_encoder_block() for _ in range(num_layers)])
    
    class Encoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
            self.norm = LayerNormalization()

        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)
    
    encoder = Encoder(layers)
    out = encoder(x.clone(), mask)

    print(f"\n--- Encoder output with {num_layers} layer(s) ---")
    print(out[0])  # print for batch index 0 only
