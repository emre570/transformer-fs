import torch.nn as nn
from residual_connection import ResidualConnection
from layer_norm import LayerNormalization

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residuals = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #1: Masked self-attention with tgt_mask
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        #2: Cross-attention with encoder_output (use src_mask)
        x = self.residuals[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        #3: FeedForward block
        x = self.residuals[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)