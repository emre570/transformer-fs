import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        #Matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        #Vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        #sin to even positions, cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # calculate variance
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear_1(x)   # 512 → 2048 (increase dimension)
        x = torch.relu(x)      # Non-linear activation
        x = self.dropout(x)    # Dropout in training
        x = self.linear_2(x)   # 2048 → 512 (decrease to original dimension)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.heads = heads

        # Define attention heads dimensions.
        assert d_model % heads == 0, "d_model is not divisible by h."
        self.head_dim = d_model // heads

        # Create linear layers for w_q, w_k, w_v. 
        # This layers does this formula:
        # Q = x @ A.T + b -> x is input vector, A.T is weighted matrix's transpose, b is bias.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Concat all weighted matrices into one with same shape.
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def calculate_attention(Q, K, V, head_dim, mask, dropout: nn.Dropout):
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)

        scores = scores.softmax(dim=-1)
        scores = dropout(scores)
        x = scores @ V
        #print("scores:", scores.shape)
        #print("mask:  ", mask.shape)
        return x

    def forward(self, q, k, v, mask=None):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Reshape matrices -> (batch, seq_len, heads, head_dim)
        # batch: m.shape[0] m is the matrix you want to give.
        # seq_len: m.shape[1]
        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.head_dim).transpose(1,2)
        K = K.view(K.shape[0], K.shape[1], self.heads, self.head_dim).transpose(1,2)
        V = V.view(V.shape[0], V.shape[1], self.heads, self.head_dim).transpose(1,2)

        # Calculate attention scores
        attention_scores = MultiHeadAttention.calculate_attention(Q, K, V, self.head_dim, mask, self.dropout)
        # 1. Transpose to (batch_size, seq_len, heads, head_dim)
        x = attention_scores.transpose(1, 2)

        # 2. Merge heads and head_dim → (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(q.shape[0], q.shape[1], self.d_model)
        x = self.w_o(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
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
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)