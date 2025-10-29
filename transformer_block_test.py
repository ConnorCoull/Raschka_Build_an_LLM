import torch
import torch.nn as nn
from utils.transformer_utils import TransformerBlock
from utils.gpt2_config import GPT_CONFIG_124M as cfg

torch.manual_seed(123)

x = torch.rand(2, 4, 768)
block = TransformerBlock(cfg)
output = block(x)

print(f"Input shape: {x.shape}")
print(f"Output Shape: {output.shape}")