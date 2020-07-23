import torch
import pickle
from tqdm import tqdm

d = torch.load('data/chat270m_clean_cache.bin')
max_len = 180
with open('data/dummy_cache', 'rb') as f:
    data = pickle.load(f)
new_tensor = None
i = 0
for d in tqdm(data):
    if len(d) <= max_len:
        data[i] = d + [0] * (max_len - len(d))
        i += 1
        tmp = torch.LongTensor(d + [0] * (max_len - len(d)))
new_tensor = torch.LongTensor(data[: i])
torch.save(new_tensor, 'data/chat270m_clean_cache.bin')
