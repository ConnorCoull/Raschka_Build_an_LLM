import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset, DataLoader):
    def __init__(self, text, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokeniser.encode(text)

        # Sliding window chunks the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + stride: i + max_length + stride]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in dataset
    def __len__(self):
        return len(self.input_ids)
    
    # Returns a row (input, target pair) from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

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