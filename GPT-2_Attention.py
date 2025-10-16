import torch
from utils.attention_utils import MultiHeadAttention

context_length = 1024
d_in, d_out = 768, 768
num_heads = 12

gpt_2_mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)

### Taken from solutions code
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The number of parameters in gpt_2_mha is: {count_parameters(gpt_2_mha)}")