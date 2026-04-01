"""
BPE Tokenizer Training Module

支持多进程预分词和高效的大文件处理。
"""

import os
import pickle # 文件操作
import time
import tracemalloc # 内存使用统计
from collections import defaultdict
from multiprocessing import Pool, cpu_count # 多进程库
from typing import BinaryIO

import regex as re
from tqdm import tqdm


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    查找文件的分块边界。

    Args:
        file: 打开的二进制文件对象。
        desired_num_chunks: 预期的分块数量
        split_special_token: 用于分块的分隔符。
    Returns:
        list[int]: 块边界位置，list[0]:第一个块的起始位置，list[-1]:最后一个块的结束位置
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 计算每个块的理想大小
    chunk_size = file_size // desired_num_chunks

    # 预设块边界
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # 设置mini chunk size的大小
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        # 从预设的块边界开始，逐个读取mini chunk
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    # 上面的方法可能导致块边界重复，因此需要去重
    # 返回块边界
    return sorted(set(chunk_boundaries))


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> list[bytes]:
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


def process_chunk_for_pre_tokenize(args: tuple) -> dict[tuple[bytes, ...], int]:
    """
    处理单个数据块的预分词。

    Args:
        args: (file_path, start, end, special_tokens)
            file_path: 输入文件路径
            start: 数据块的起始位置
            end: 数据块的结束位置
            special_tokens: 预分词中的特殊标记列表
    Returns:
        该块的token频率字典
    """
    file_path, start, end, special_tokens = args
    token_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    # 每次读取 1MB 缓冲区，安全不爆内存
    BUFFER_SIZE = 1024 * 1024
    special_token_bytes = b"<|endoftext|>"
    token_buffer = b""
    # 读取二进制数据
    with open(file_path, 'rb') as f:
        f.seek(start)
        current_pos = f.tell()
        # 如果不是开头，需要找到下一个<|endoftext|>之后的位置
        if start > 0:
            # 读取一小段找到分隔符
            temp_buf = f.read(min(1024, end - current_pos))
            eot_pos = temp_buf.find(special_token_bytes)
            if eot_pos != -1:
                f.seek(start + eot_pos + len(special_token_bytes))
            else:
                # 没找到分隔符，从start开始
                f.seek(start)
            current_pos = f.tell()

        while current_pos < end:
            read_size = min(BUFFER_SIZE, end - current_pos)
            data = f.read(read_size)
            if not data:
                break

            token_buffer += data
            current_pos = f.tell()

            # 按文档分隔符切割处理
            while special_token_bytes in token_buffer:
                doc_part, token_buffer = token_buffer.split(special_token_bytes, 1)
                if not doc_part:
                    continue

                # 解码
                try:
                    text = doc_part.decode('utf-8', errors='ignore')
                except:
                    continue

                if not text:
                    continue

                # 预分词
                try:
                    pre_tokens = pre_tokenize(text, special_tokens)
                except:
                    continue

                # 统计频率
                for token in pre_tokens:
                    token_tuple = tuple(bytes([b]) for b in token)
                    token_freqs[token_tuple] += 1

        # 处理缓冲区剩余内容
        if token_buffer:
            try:
                text = token_buffer.decode('utf-8', errors='ignore')
                if text:
                    pre_tokens = pre_tokenize(text, special_tokens)
                    for token in pre_tokens:
                        token_tuple = tuple(bytes([b]) for b in token)
                        token_freqs[token_tuple] += 1
            except:
                pass

    return token_freqs


def merge_token_frequencies(freq_list: list[dict[tuple[bytes, ...], int]]) -> dict[tuple[bytes, ...], int]:
    """
    合并多个频率字典。

    Args:
        freq_list: 频率字典列表。

    Returns:
        合并后的频率字典。
    """
    merged: dict[tuple[bytes, ...], int] = defaultdict(int)
    for freqs in freq_list:
        for token, count in freqs.items():
            merged[token] += count
    return dict(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个BPE tokenizer。

    Args:
        input_path: 包含训练文本的文件路径。
        vocab_size: 词汇表的目标大小。
        special_tokens: 可选的特殊标记列表，这些标记将直接添加到词汇表中，并且在训练过程中不会被合并。
        num_workers: 并行工作进程数，默认为CPU核心数。

    Returns:
        vocab: 把token id 映射到token bytes的字典。
        merges: 合并的规则列表。
    """
    if special_tokens is None:
        special_tokens = []

    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 限制最大进程数

    # 阶段1: 多进程预分词
    with open(input_path, 'rb') as f:
        file_size = os.fstat(f.fileno()).st_size

        # 计算分块边界
        chunk_boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

    # 准备进程参数
    process_args = []
    for i in range(len(chunk_boundaries) - 1):
        process_args.append((
            str(input_path),
            chunk_boundaries[i],
            chunk_boundaries[i + 1],
            special_tokens
        ))

    # 并行处理
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_chunk_for_pre_tokenize, process_args),
            total=len(process_args),
            desc="预分词处理"
        ))

    # 合并频率统计
    token_freqs = merge_token_frequencies(results)

    # 初始化词表，256个字节
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

    # 添加特殊标记到词汇表
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    # 执行BPE合并
    with tqdm(total=vocab_size - len(vocab), desc="BPE合并") as pbar:
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

            # Clear the best pair's tracking
            pair_to_tokens[best_pair].clear()
            pair_counts[best_pair] = 0

            pbar.update(1)

    return vocab, merges


def main():
    """主函数：训练BPE分词器并保存结果。"""
    import argparse

    parser = argparse.ArgumentParser(description="训练BPE分词器")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--vocab-size", type=int, default=10000, help="词汇表大小")
    parser.add_argument("--output-dir", type=str, default="./tokenizer", help="输出目录")
    parser.add_argument("--output-prefix", type=str, default="bpe", help="输出文件前缀")
    parser.add_argument("--num-workers", type=int, default=None, help="并行工作进程数")
    parser.add_argument("--profile", action="store_true", help="启用性能分析")

    args = parser.parse_args()

    # 特殊标记
    special_tokens = ["<|endoftext|>"]

    print(f"开始训练BPE分词器...")
    print(f"输入文件: {args.input}")
    print(f"词汇表大小: {args.vocab_size}")
    print(f"特殊标记: {special_tokens}")

    # 开始性能分析
    if args.profile:
        tracemalloc.start()

    start_time = time.time()

    # 训练BPE
    vocab, merges = train_bpe(
        args.input,
        args.vocab_size,
        special_tokens,
        num_workers=args.num_workers,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # 获取内存使用情况
    if args.profile:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\n性能统计:")
        print(f"  训练耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
        print(f"  当前内存使用: {current / 1024 / 1024:.2f} MB")
        print(f"  峰值内存使用: {peak / 1024 / 1024:.2f} MB ({peak / 1024 / 1024 / 1024:.2f} GB)")
    else:
        print(f"\n训练耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

    # 保存结果
    output_dir = args.output_dir
    output_prefix = args.output_prefix
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, output_prefix + "_vocab.pkl")
    merges_path = os.path.join(output_dir, output_prefix + "_merges.pkl")

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"词汇表已保存到: {vocab_path}")

    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
    print(f"合并规则已保存到: {merges_path}")

    # 分析词汇表
    print(f"\n词汇表统计:")
    print(f"  总词汇量: {len(vocab)}")

    # 找出最长的标记
    max_len = 0
    longest_tokens = []
    for token_id, token_bytes in vocab.items():
        token_len = len(token_bytes)
        if token_len > max_len:
            max_len = token_len
            longest_tokens = [(token_id, token_bytes)]
        elif token_len == max_len:
            longest_tokens.append((token_id, token_bytes))

    print(f"  最长标记长度: {max_len} 字节")
    print(f"  最长标记数量: {len(longest_tokens)}")
    print(f"  最长标记示例:")
    for token_id, token_bytes in longest_tokens[:5]:
        try:
            decoded = token_bytes.decode('utf-8', errors='replace')
            print(f"    ID {token_id}: {repr(decoded)} ({len(token_bytes)} bytes)")
        except:
            print(f"    ID {token_id}: {token_bytes!r} ({len(token_bytes)} bytes)")

    # 分析合并规则
    print(f"\n合并规则统计:")
    print(f"  总合并规则数: {len(merges)}")


if __name__ == "__main__":
    main()
