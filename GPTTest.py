import torch
import torch.nn
from utils.GPTModel import GPTModel
from utils.gpt2_config import GPT_CONFIG_124M as cfg
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch.shape)

torch.manual_seed(123)
model = GPTModel(cfg)

out = model(batch)

print(f"Input: {batch}")
print(f"Output Shape: {out.shape}")
print(f"Output: {out}")