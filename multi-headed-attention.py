import torch
from utils.attention_utils import MultiHeadAttentionWrapper
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]
    ]
)

torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
d_in, d_out = 3, 2
context_length = batch.shape[1]

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)

print("context_vecs.shape:", context_vecs.shape)
