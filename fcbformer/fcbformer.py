import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.linear = nn.Linear(hidden_size, input_size)
        self.num_layers = num_layers

    def forward(self, x):
        x = self.embedding(x)
        for i in range(self.num_layers):
            attn_output, _ = self.multihead_attn(x, x, x)
            x = F.layer_norm(x + attn_output, [1, 2])
            ff_output = self.feedforward(x)
            x = F.layer_norm(x + ff_output, [1, 2])
        x = self.linear(x)
        return x