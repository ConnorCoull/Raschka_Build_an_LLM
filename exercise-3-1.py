# Hoenst check - had to cheat to figure this one out a bit
# I understood the idea of moving the values from v2 to v1 but thought
# It was a case of saving a model and loading the values into v1

# Issue 1 - how to move values by calling v1 = v2.weight.T
# Issue 2 - torch.nn.parameter
# Issue 3 - torch.nn.Parameter

import torch
from utils.attention_utils import SelfAttention_v1, SelfAttention_v2

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

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v1 = SelfAttention_v1(d_in, d_out)

sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

# Verfied to be identical
print(sa_v1(inputs))
print(sa_v2(inputs))