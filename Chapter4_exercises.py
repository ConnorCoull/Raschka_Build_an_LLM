from utils.GPTModel import GPTModel
from utils.gpt2_config import GPT_CONFIG_124M as cfg
from utils.transformer_utils import TransformerBlock

#gpt = GPTModel(cfg)

#print(gpt.transformer_blocks.ff.weight.shape)
# can't get to ff through sequential

transformer_block = TransformerBlock(cfg)
#print(transformer_block.ff.weight.shape)
# ff has no weights, use params and numel


total_tbff_params = (
    sum(p.numel() for p in transformer_block.ff.parameters())
)

total_tbatt_params = (
    sum(p.numel() for p in transformer_block.att.parameters())
)

print(total_tbff_params)
print(total_tbatt_params)