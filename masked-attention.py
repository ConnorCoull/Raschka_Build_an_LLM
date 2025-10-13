import torch
from utils.attention_utils import SelfAttention_v2

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]
    ]
)

torch.manual_seed(789)

d_in = inputs.shape[1]
d_out = 2

sa_v2 = SelfAttention_v2(d_in, d_out)

# AttributeError: type object 'SelfAttention_v2' has no attribute 'W_query'
# forgot d_in d_out in sa_v2 init
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

masked_simple = attn_weights * mask_simple
# print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)