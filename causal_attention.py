import torch
from utils.attention_utils import CausalAttention
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]
    ]
)

# x_T = inputs[T-1]
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
batch = torch.stack((inputs, inputs), dim=0)
print("batch.shape: ", batch.shape)

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape: ", context_vecs.shape)