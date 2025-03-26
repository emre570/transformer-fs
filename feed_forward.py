import torch
import torch.nn as nn

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