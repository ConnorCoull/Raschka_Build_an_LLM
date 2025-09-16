import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset, DataLoader):
    def __init__(self, text, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokeniser.encode(text)

        # Sliding window chunks the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in dataset
    def __len__(self):
        return len(self.input_ids)
    
    # Returns a row (input, target pair) from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]