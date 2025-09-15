import re
from tokeniser_utils import SimpleTokeniserV1, SimpleTokeniserV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"The total number of characters is {len(raw_text)}.\n")
print(f"The first 100 characters are:\n{raw_text[:100]}\n")

preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
print(f"Length of the pre-processed text: {len(preprocessed_text)}")

all_tokens = set(preprocessed_text)
vocab_size = len(all_tokens)
print(f"The vocabulary size is {vocab_size}.\n")

vocab = {token: integer for integer, token in enumerate(sorted(all_tokens))}

# Print first 50
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 49:
#         break

tokeniser = SimpleTokeniserV1(vocab)
text = """It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."""
ids = tokeniser.encode(text)
print(ids)

print(tokeniser.decode(ids))

all_tokens = sorted(list(set(preprocessed_text)))
all_tokens.extend(["<UNK>", "<ENDOFTEXT>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

text1 = "Hello, do you like Python?"
text2 = "In the sunlit terraces of the palace."
text = " <ENDOFTEXT> ".join([text1, text2])
print(text)

tokeniser = SimpleTokeniserV2(vocab)
print(tokeniser.encode(text))
print(tokeniser.decode(tokeniser.encode(text)))