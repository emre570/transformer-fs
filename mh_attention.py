import torch.nn as nn
import math

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