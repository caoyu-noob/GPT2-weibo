from tqdm import tqdm

input_file = 'data/train.txt'
output_file = 'data/train_clean.txt'

def check_string(string):
    symbol_cnt = 0
    for c in string:
        if (ord(c) >= 65 and ord(c) <= 90) or (ord(c) >= 97 and ord(c) <= 122) or \
                (c in ['(', ')', '.', ',', ':', '?', '/', '\\', '\'', '\"', '[', ']', '{', '}', '+', '（', '）', '《', '》'
                    , '。', '，', '；', '：', '‘', '’', '“', '”', ' ']) or c.isdigit():
            symbol_cnt += 1
    if len(string) - symbol_cnt <= 1:
        return False
    return True

with open(input_file, 'r', encoding='utf-8') as f:
    raw_lines = f.readlines()
i = 0
ori_len = len(raw_lines)
for raw_line in tqdm(raw_lines):
    items = raw_line.strip().split('\t')
    if len(items) < 2:
        continue
    q = items[0].strip()
    r = items[1].strip()
    if check_string(q) and check_string(r):
        raw_lines[i] = '\t'.join([q, r]) + '\n'
        i += 1
print('Before clean: ' + str(ori_len))
print('After clean: ' + str(i))
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(raw_lines[:i])