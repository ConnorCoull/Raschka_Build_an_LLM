import torch
from utils.activation_function_utils import FeedForward
from utils.gpt2_config import GPT_CONFIG_124M

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, GPT_CONFIG_124M["emb_dim"])
out = ffn(x)
print(out.shape)
