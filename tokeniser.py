import re
import torch
import tiktoken
from utils.dataloader_utils import GPTDatasetV1
from torch.utils.data import DataLoader

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokeniser = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokeniser, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

downloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=4, shuffle=False)
data_iter = iter(downloader)
first_batch = next(data_iter)
print(first_batch)

