from utils.dataloader_utils import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

downloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=4, shuffle=False)
data_iter = iter(downloader)
first_batch = next(data_iter)
print(first_batch)

