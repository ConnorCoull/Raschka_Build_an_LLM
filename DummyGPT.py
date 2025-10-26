import torch
import tiktoken
from utils.gpt2_config import GPT_CONFIG_124M
from utils.DummyGPTModel import DummyGPTModel

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch.shape)

model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape: ", logits.shape)
print(logits)
# Returns the batch shape plus an extra vocab size len embedding
