import os
from typing import BinaryIO
import regex as re
from collections import defaultdict


def pre_tokenize(text: str, special_tokens: list[str] | None) -> list[bytes]:
    """
    预分词，返回字节串列表。

    Args:
        text: 输入文本
        special_tokens: 特殊标记列表
    Returns:
        list[bytes]: 字节串列表
    """
    # GPT2风格的分词规则
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if not special_tokens:
        out = re.findall(pattern, text)
        return [bytes(b.encode("utf-8")) for b in out]

    toks = sorted(special_tokens, key=len, reverse=True)
    union = "|".join(re.escape(t) for t in toks)
    parts = re.split(f"({union})", text)

    out = []
    st = set(special_tokens)
    for part in parts:
        # 空串，跳过
        if not part:
            continue
        # special 只作为边界，完全跳过
        if part in st:
            continue
        out.extend(re.findall(pattern, part))
    return [bytes(b.encode("utf-8")) for b in out]



def get_initial_vocab() -> dict[int, bytes]:
    """
    获取初始化词汇表（size: 256）。

    Return:
        dict[int, bytes]: 初始词汇表
    """
    return {i: bytes([i]) for i in range(256)}


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个BPE tokenizer。

    Args:
        input_path: 包含训练文本的文件路径。
        vocab_size: 词汇表的目标大小。
        special_tokens: 可选的特殊标记列表，这些标记将直接添加到词汇表中，并且在训练过程中不会被合并。

    Returns:
        vocab: 把token id 映射到token bytes的字典。
        merges: 合并的规则列表。
    """
    if special_tokens is None:
        special_tokens = []

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 预分词，得到词块记为T1，通常是一个词汇或短句的对应的字节串。
    pre_tokens = pre_tokenize(text, special_tokens)

    # 统计预分词得到的T1的出现频率
    token_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    for token in pre_tokens:
        token_tuple = tuple(bytes([b]) for b in token)
        token_freqs[token_tuple] += 1

    # 初始化词表，256个字节，词表中的token记为T2类型，也就是说，初始时一个T1包含若干个T2
    vocab = get_initial_vocab()
    # 初始化合并规则表
    merges: list[tuple[bytes, bytes]] = []

    # 初始化T2对的频率
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # 追踪T2对关联的T1
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    # 初始化T2对的频率
    for token, freq in token_freqs.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += freq
            pair_to_tokens[pair].add(token)

    # 添加特殊标记
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    # 执行BPE合并
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 找到最高频的T2对
        # 先比较频率，再比较字节字典序
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

        if pair_counts[best_pair] == 0:
            break

        # 增加合并规则，更新词表
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token

        # 获取受影响的T1
        affected_tokens = list(pair_to_tokens[best_pair])

        # 更新
        for old_token in affected_tokens:
            if old_token not in token_freqs:
                continue

            freq = token_freqs[old_token]

            # 移除旧T1对pair_counts的贡献
            for i in range(len(old_token) - 1):
                pair = (old_token[i], old_token[i + 1])
                pair_counts[pair] -= freq
                if old_token in pair_to_tokens[pair]:
                    pair_to_tokens[pair].discard(old_token)

            # 生成新T2
            new_token_list = []
            i = 0
            while i < len(old_token):
                if i < len(old_token) - 1 and old_token[i] == best_pair[0] and old_token[i + 1] == best_pair[1]:
                    new_token_list.append(new_token)
                    i += 2
                else:
                    new_token_list.append(old_token[i])
                    i += 1

            new_token_tuple = tuple(new_token_list)

            # 更新freq
            del token_freqs[old_token]
            token_freqs[new_token_tuple] = freq

            # 增加新T1对pair_counts的贡献
            for j in range(len(new_token_tuple) - 1):
                pair = (new_token_tuple[j], new_token_tuple[j + 1])
                pair_counts[pair] += freq
                pair_to_tokens[pair].add(new_token_tuple)

        pair_to_tokens[best_pair].clear()
        pair_counts[best_pair] = 0

    return vocab, merges
