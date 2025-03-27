import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 heads: int,
                 dropout: float):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        
        # TODO: Define input/output embeddings
        # TODO: Positional encoding
        # TODO: Encoder stack
        # TODO: Decoder stack
        # TODO: Final linear layer (project to tgt_vocab_size)
