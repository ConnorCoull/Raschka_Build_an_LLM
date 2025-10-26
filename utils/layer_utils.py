import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # Some small constant added to varience to prevent division by 0 err
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Trainable param
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # Trainable param

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift