import torch
import torch.nn as nn
from utils.layer_utils import LayerNorm

torch.manual_seed(123)
batch = torch.randn(2, 5)
#print(batch)
# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch)
#print(out)

# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True)
#print(f"Mean: {mean}")
#print(f"Var: {var}")

# out_norm = (out - mean)/ torch.sqrt(var)
# print(f"Normalised layer outputs: {out_norm}")
torch.set_printoptions(sci_mode=False)
# mean = out_norm.mean(dim=-1, keepdim=True)
# var = out_norm.var(dim=-1, keepdim=True)
# print(f"Mean: {mean}")
# print(f"Var: {var}")

# Bug - basically different rounding for tiny floating point errors, but effectively 0 mean and 1 var for both

# With new layer norm
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print(f"Mean: {mean}")
print(f"Var: {var}")