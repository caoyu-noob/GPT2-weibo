import pickle
from tqdm import tqdm

with open('data/inference.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
print('111')

with open('data/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
all_dict = {}
for line in tqdm(lines):
    items = line.split('\t')
    if len(items) < 2:
        continue
    q = items[0].strip()
    if not all_dict.__contains__(q):
        all_dict[q] = []
    all_dict[q].append(items[1].strip())
lines = []
for k, v in all_dict.items():
    lines.append(k + '\n')
with open('data/inference.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines)
with open('data/responses.pickle', 'wb') as f:
    pickle.dump(all_dict, f)
print('111')