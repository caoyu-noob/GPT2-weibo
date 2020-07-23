import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from train import create_model
import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0


def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='对话模型路径')
    parser.add_argument('--inference_path', type=str, required=True, help='inference文件路径')
    parser.add_argument('--inference_results_path', type=str, default='inference_results.txt', help='inference结果文件路径')
    parser.add_argument('--inference_mode', type=str, default="sample", help='inference的方法，sample/beam')
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='beam search模式下beam大小')
    parser.add_argument('--voca_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def config_logger(log_path):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger

# def load_inference_data(args, tokenizer, max_len=512):
#     with open(args.inference_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     data = []
#     for line in lines:
#         line = line.strip()
#         ids = [tokenizer.cls_token_id]
#         ids.extend([tokenizer.convert_tokens_to_ids(word) for word in line])
#         ids.append(tokenizer.sep_token_id)
#         ids = ids[:max_len]
#         data.append(ids)
#     return data

def load_inference_data(args, tokenizer, max_len=512):
    with open(args.inference_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        ids = [tokenizer.cls_token_id]
        ids.extend(tokenizer.encode(line)[1:-1])
        ids.append(tokenizer.sep_token_id)
        ids = ids[:max_len]
        data.append(ids)
    return data

def sample_inference(args, model, tokenizer, sample, device):
    input_ids = torch.tensor(sample, dtype=torch.long).to(device)
    generated = []
    with torch.no_grad():
        for _ in range(args.max_len):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs[0][-1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(generated):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            generated.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token), dim=0)
    return generated

def _get_proba_with_temperature(logits, bs_temperature=1):
    if bs_temperature != 1:
        logits /= bs_temperature

    return torch.nn.functional.softmax(logits, dim=-1)

def _get_beam_scores(probas, beam_scores, is_end, bs_nucleus_p=0, annealing_topk=None):
    skip_mask = None

    if bs_nucleus_p > 0:
        assert annealing_topk is None

        sorted_probas, idxs = torch.sort(probas, descending=True, dim=-1)
        skip_mask = torch.cumsum(sorted_probas.cumsum(dim=-1) > bs_nucleus_p, dim=-1) > 1
        sorted_probas.masked_fill_(skip_mask, 0.0)
        _, idxs = torch.sort(idxs, dim=-1)
        probas = torch.gather(sorted_probas, -1, idxs)
        skip_mask = torch.gather(skip_mask, -1, idxs)
    beam_scores = beam_scores.unsqueeze(-1) + torch.log(probas + 1e-20) * (1 - is_end.float().unsqueeze(-1))

    if skip_mask is not None:
        beam_scores.masked_fill_(skip_mask, float('-inf'))

    return beam_scores

def _sample(beam_scores, num_samples, sample_prob=1., annealing_topk=None):
    if random.random() < sample_prob:
        beam_probas = torch.nn.functional.softmax(beam_scores, dim=-1)
        if annealing_topk is not None:
            beam_probas, sample_idxs = beam_probas.topk(annealing_topk, dim=-1)
            idxs = torch.multinomial(beam_probas, num_samples)
            idxs = torch.gather(sample_idxs, 1, idxs)
        else:
            idxs = torch.multinomial(beam_probas, num_samples)
        scores = torch.gather(beam_scores, 1, idxs)
    else:
        scores, idxs = beam_scores.topk(num_samples, dim=-1)

    return scores, idxs

def _fix_past(past, beam_idxs, beam_size):
    for layer_output in past:
        for v in layer_output:
            size_ = v.size()
            tile_size = size_[-2] * size_[-1] * size_[-3]
            new_v = v.contiguous().view(-1, beam_size, tile_size)
            new_v = new_v.gather(1, beam_idxs.unsqueeze(-1).repeat([1, 1, tile_size]))
            v[...] = new_v.view(*size_)
    return past

def _length_penalty(sequence_lengths, length_penalty_coef=0.6):
    """https://arxiv.org/abs/1609.08144"""
    return (5 + sequence_lengths) ** length_penalty_coef / (5 + 1) ** length_penalty_coef

def beam_inference(args, model, tokenizer, sample, device, diversity_groups=1, batch_size=1, current_sample_prob=1,
                   annealing=1):
    n_embeddings = tokenizer.vocab_size
    input_ids = torch.tensor(sample, dtype=torch.long).to(device)
    beam_scores = torch.zeros(1, args.beam_size, device=device)
    beam_lens = torch.ones(1, args.beam_size, dtype=torch.long, device=device)
    is_end = torch.zeros(1, args.beam_size, dtype=torch.uint8, device=device)
    diversity_penalty = torch.zeros((batch_size, n_embeddings), device=device)
    prevs = torch.full((batch_size * args.beam_size, 1), fill_value=tokenizer.sep_token_id, dtype=torch.long,
                               device=device)
    past = None
    group_size = args.beam_size // diversity_groups
    for i in range(args.max_len):
        if i == 0:
            input_ids = input_ids.unsqueeze(0)
            input_ids = torch.cat([input_ids] * args.beam_size, dim=0)
            lm_logits, past = model(input_ids)
            lm_logits = lm_logits[:, -1:, :]
        else:
            inputs_ids = prevs[:, -1:, ...]  # only use the last token (rest is in past)
            lm_logits, past = model(inputs_ids, past=past)
        probs = _get_proba_with_temperature(lm_logits.float())
        probs = probs.view(batch_size, args.beam_size, -1)

        beam_scores = _get_beam_scores(probs, beam_scores, is_end)
        penalty = _length_penalty(beam_lens.float() + 1 - is_end.float()).unsqueeze(-1)
        beam_scores = beam_scores / penalty

        if i == 0:
            penalty = penalty[:, 0, :]
            beam_scores = beam_scores[:, 0, :]

            beam_scores, idxs = beam_scores.topk(args.beam_size, dim=-1)
            beam_idxs = torch.zeros((batch_size, args.beam_size), dtype=torch.long, device=device)
        else:
            penalty = penalty.view(batch_size, diversity_groups, group_size, -1)
            beam_scores = beam_scores.view(1, diversity_groups, group_size, -1)

            all_scores, all_idxs = [], []
            for g in range(diversity_groups):
                g_beam_scores = beam_scores[:, g, :, :]
                g_beam_scores = g_beam_scores.view(batch_size, -1)

                g_scores, g_idxs = _sample(g_beam_scores, group_size, sample_prob=current_sample_prob)
                g_idxs += g * group_size * n_embeddings

                all_scores.append(g_scores)
                all_idxs.append(g_idxs)

                diversity_penalty.scatter_add_(1,
                                               torch.fmod(g_idxs, n_embeddings),
                                               torch.ones((batch_size, group_size), device=device))

            diversity_penalty.fill_(0)
            penalty = penalty.view(batch_size, -1)
            beam_scores = torch.cat(all_scores, dim=-1)
            idxs = torch.cat(all_idxs, dim=-1)

            beam_idxs = (idxs.float() / n_embeddings).long()

        sym_idxs = torch.fmod(idxs, probs.shape[-1])
        is_end = torch.gather(is_end, 1, beam_idxs)
        beam_lens = torch.gather(beam_lens, 1, beam_idxs)

        sym_idxs[is_end] = tokenizer.pad_token_id

        beam_lens[~is_end] += 1
        is_end[sym_idxs == tokenizer.sep_token_id] = 1

        sym_idxs = sym_idxs.view(1 * args.beam_size, 1)
        prevs = prevs.view(1, args.beam_size, -1)
        prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
        prevs = prevs.view(1 * args.beam_size, -1)
        prevs = torch.cat([prevs, sym_idxs], dim=1)

        past = _fix_past(past, beam_idxs, args.beam_size)

        if all(is_end.view(-1)):
            break

        beam_scores *= penalty
        current_sample_prob *= annealing

    result = prevs.view(batch_size, args.beam_size, -1)
    bests = beam_scores.argmax(dim=-1)
    best_len = beam_lens[0, bests[0]]
    best_seq = result[0, bests[0], :best_len - 1]
    generated = best_seq.tolist()
    return generated

def main():
    args = set_interact_args()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(args.model_path, 'inference-{}.log'.format(current_time))
    logger = config_logger(logdir)

    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    logger.info('Loading inference data')
    inference_data = load_inference_data(args, tokenizer)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    all_inference_results = []
    for i, sample in enumerate(tqdm(inference_data)):
        if args.inference_mode == 'sample':
            generated = sample_inference(args, model, tokenizer, sample, device)
        elif args.inference_mode == 'beam':
            generated = beam_inference(args, model, tokenizer, sample, device)
        text = ''.join(tokenizer.convert_ids_to_tokens(generated, skip_special_tokens=True))
        q_text = ''.join(tokenizer.convert_ids_to_tokens(sample, skip_special_tokens=True))
        all_inference_results.append(q_text + ' | ' + text + '\n')
        if (i + 1) % 1000 == 0:
            logger.info('q: ' + q_text + '  r: ' + text)

    with open(os.path.join(args.model_path, args.inference_results_path), 'w', encoding='utf-8') as f:
        f.writelines(all_inference_results)

if __name__ == '__main__':
    main()
