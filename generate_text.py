import torch
from utils.GPTModel import GPTModel
from utils.gpt2_config import GPT_CONFIG_124M as cfg
import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = GPTModel(cfg)

start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
print(f"encoded: {encoded}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(f"encoded tensor shape: {encoded_tensor.shape}")

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=cfg["context_length"]
)

print(f"Output: {out}")
print(f"Output length: {len(out[0])}")

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"Decoded Text: {decoded_text}")