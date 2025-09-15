import re

class SimpleTokeniserV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        # The inconsistency between token, item, and s is mostly my own fault
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_text = [token.strip() for token in preprocessed_text if token.strip()]
        ids = [self.str_to_int[token] for token in preprocessed_text]
        return ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        # Removes space before punctuation
        text = re.sub(r'\s([,.:;?_!"\'])', r'\1', text)
        return text
    
class SimpleTokeniserV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        # The inconsistency between token, item, and s is mostly my own fault
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_text = [token.strip() for token in preprocessed_text if token.strip()]
        preprocessed_text = [token if token in self.str_to_int else "<UNK>" for token in preprocessed_text]
        ids = [self.str_to_int[token] for token in preprocessed_text]
        return ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        # Removes space before punctuation
        text = re.sub(r'\s([,.:;?_!"\'])', r'\1', text)
        return text