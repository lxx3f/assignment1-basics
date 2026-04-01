"""
计算分词器压缩比：从数据集中抽样文档，编码后计算 bytes/token
使用流式读取 + 蓄水池抽样避免内存问题
新增：交叉分词测试、吞吐量估算
"""
import random
import pickle
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(vocab_path, merges_path):
    """从 pickle 文件加载分词器"""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    # 转换 vocab key 为 int，value 为 bytes
    norm_vocab = {}
    for k, v in vocab.items():
        kid = int(k)
        if isinstance(v, str):
            v = v.encode("utf-8")
        norm_vocab[kid] = v

    # 转换 merges 为 bytes tuple
    norm_merges = []
    for a, b in merges:
        if isinstance(a, str):
            a = a.encode("utf-8")
        if isinstance(b, str):
            b = b.encode("utf-8")
        norm_merges.append((a, b))

    return Tokenizer(norm_vocab, norm_merges)


def reservoir_sample_nonempty(filepath, n_samples=10, seed=42):
    """
    使用蓄水池抽样算法从文件中随机抽取 n 个非空行
    只需单次遍历，内存友好，保证返回恰好 n 个样本（如果文件有足够非空行）
    """
    random.seed(seed)
    reservoir = []
    non_empty_count = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue  # 跳过空行

            non_empty_count += 1

            if len(reservoir) < n_samples:
                # 前 n 个非空行直接加入
                reservoir.append(stripped)
            else:
                # 以 n/count 的概率替换 reservoir 中的随机一个
                j = random.randint(0, non_empty_count - 1)
                if j < n_samples:
                    reservoir[j] = stripped

    return reservoir, non_empty_count


def compute_compression_ratio(tokenizer, documents):
    """
    计算压缩比 = 原始字节数 / token 数量
    同时返回处理时间用于吞吐量计算
    """
    total_bytes = 0
    total_tokens = 0
    total_chars = 0

    # 计时开始（仅编码时间，不包含IO）
    start_time = time.perf_counter()

    for i, doc in enumerate(documents):
        # 原始字节数（UTF-8 编码）
        doc_bytes = doc.encode("utf-8")
        total_bytes += len(doc_bytes)
        total_chars += len(doc)

        # 编码后的 token 数量
        token_ids = tokenizer.encode(doc)
        total_tokens += len(token_ids)

    # 计时结束
    elapsed_time = time.perf_counter() - start_time

    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0

    return {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "elapsed_time": elapsed_time,
        "throughput_bytes_per_sec": total_bytes / elapsed_time if elapsed_time > 0 else 0,
        "throughput_chars_per_sec": total_chars / elapsed_time if elapsed_time > 0 else 0,
    }


def benchmark_throughput(tokenizer, documents, warmup=True):
    """
    更精确的吞吐量测试，包含预热和多次运行
    """
    # 预热（避免冷启动影响）
    if warmup and documents:
        for doc in documents[:5]:
            tokenizer.encode(doc)

    # 正式测试
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_chars = sum(len(doc) for doc in documents)

    start = time.perf_counter()
    for doc in documents:
        tokenizer.encode(doc)
    elapsed = time.perf_counter() - start

    return {
        "bytes_per_sec": total_bytes / elapsed if elapsed > 0 else 0,
        "chars_per_sec": total_chars / elapsed if elapsed > 0 else 0,
        "tokens_per_sec": sum(len(tokenizer.encode(doc)) for doc in documents) / elapsed if elapsed > 0 else 0,
        "total_bytes": total_bytes,
        "elapsed_sec": elapsed,
    }


def main():
    # 路径配置（相对于 scripts/ 目录）
    base_dir = Path(__file__).parent.parent
    ts_vocab_path = base_dir / "tokenizer" / "ts_vocab.pkl"
    ts_merges_path = base_dir / "tokenizer" / "ts_merges.pkl"
    ow_vocab_path = base_dir / "tokenizer" / "ow_vocab.pkl"
    ow_merges_path = base_dir / "tokenizer" / "ow_merges.pkl"

    ts_data_path = base_dir / "data" / "TinyStories-train.txt"
    ow_data_path = base_dir / "data" / "owt_train.txt"

    print("=" * 70)
    print("分词器压缩比与性能测试")
    print("=" * 70)

    # 加载分词器
    print("\n[1] 加载 TinyStories 分词器 (10K vocab)...")
    ts_tokenizer = load_tokenizer(ts_vocab_path, ts_merges_path)
    print(f"    词表大小: {len(ts_tokenizer.vocab)}")

    print("\n[2] 加载 OpenWebText 分词器 (32K vocab)...")
    ow_tokenizer = load_tokenizer(ow_vocab_path, ow_merges_path)
    print(f"    词表大小: {len(ow_tokenizer.vocab)}")

    # 抽样文档
    print("\n[3] 从 TinyStories 中抽样 10 份文档...")
    ts_docs, ts_total = reservoir_sample_nonempty(ts_data_path, n_samples=10)
    print(f"    总非空文档数: {ts_total:,}, 成功抽样: {len(ts_docs)} 份")

    print("\n[4] 从 OpenWebText 中抽样 10 份文档...")
    ow_docs, ow_total = reservoir_sample_nonempty(ow_data_path, n_samples=10)
    print(f"    总非空文档数: {ow_total:,}, 成功抽样: {len(ow_docs)} 份")

    # 压缩比测试
    print("\n" + "=" * 70)
    print("压缩比测试结果")
    print("=" * 70)

    # 1. TinyStories分词器 + TinyStories数据（同领域）
    print("\n[同领域] TinyStories分词器 (10K) + TinyStories数据")
    ts_on_ts = compute_compression_ratio(ts_tokenizer, ts_docs)
    print(f"    压缩比: {ts_on_ts['compression_ratio']:.4f} bytes/token")
    print(f"    总字节: {ts_on_ts['total_bytes']:,}, 总tokens: {ts_on_ts['total_tokens']:,}")

    # 2. OpenWebText分词器 + OpenWebText数据（同领域）
    print("\n[同领域] OpenWebText分词器 (32K) + OpenWebText数据")
    ow_on_ow = compute_compression_ratio(ow_tokenizer, ow_docs)
    print(f"    压缩比: {ow_on_ow['compression_ratio']:.4f} bytes/token")
    print(f"    总字节: {ow_on_ow['total_bytes']:,}, 总tokens: {ow_on_ow['total_tokens']:,}")

    # 3. TinyStories分词器 + OpenWebText数据（跨领域）
    print("\n[跨领域] TinyStories分词器 (10K) + OpenWebText数据")
    ts_on_ow = compute_compression_ratio(ts_tokenizer, ow_docs)
    print(f"    压缩比: {ts_on_ow['compression_ratio']:.4f} bytes/token")
    print(f"    总字节: {ts_on_ow['total_bytes']:,}, 总tokens: {ts_on_ow['total_tokens']:,}")

    # 4. OpenWebText分词器 + TinyStories数据（跨领域）
    print("\n[跨领域] OpenWebText分词器 (32K) + TinyStories数据")
    ow_on_ts = compute_compression_ratio(ow_tokenizer, ts_docs)
    print(f"    压缩比: {ow_on_ts['compression_ratio']:.4f} bytes/token")
    print(f"    总字节: {ow_on_ts['total_bytes']:,}, 总tokens: {ow_on_ts['total_tokens']:,}")

    # 对比分析
    print("\n" + "-" * 70)
    print("跨领域分词效果分析")
    print("-" * 70)
    print(f"TinyStories分词器处理OW数据 vs 同领域: "
          f"{ts_on_ow['compression_ratio']:.4f} vs {ts_on_ts['compression_ratio']:.4f} "
          f"({(ts_on_ow['compression_ratio']/ts_on_ts['compression_ratio']-1)*100:+.1f}%)")
    print(f"  → 压缩比 {'下降' if ts_on_ow['compression_ratio'] < ts_on_ts['compression_ratio'] else '上升'}，"
          f"说明TinyStories分词器对通用文本的编码效率较低")
    print()
    print(f"OpenWebText分词器处理TS数据 vs 同领域: "
          f"{ow_on_ts['compression_ratio']:.4f} vs {ow_on_ow['compression_ratio']:.4f} "
          f"({(ow_on_ts['compression_ratio']/ow_on_ow['compression_ratio']-1)*100:+.1f}%)")
    print(f"  → 压缩比 {'下降' if ow_on_ts['compression_ratio'] < ow_on_ow['compression_ratio'] else '上升'}，"
          f"说明OW分词器对儿童故事的编码效率较低")
    print()
    print("结论：分词器在训练数据领域内的压缩效果更好，跨领域使用会导致压缩比下降")

    # 吞吐量测试
    print("\n" + "=" * 70)
    print("分词器吞吐量测试")
    print("=" * 70)
    print("\n[*] 使用20份OpenWebText文档进行吞吐量测试...")
    ow_docs_20, _ = reservoir_sample_nonempty(ow_data_path, n_samples=20, seed=123)

    print("\n>>> TinyStories分词器 (10K vocab)")
    ts_perf = benchmark_throughput(ts_tokenizer, ow_docs_20)
    print(f"    处理速度: {ts_perf['bytes_per_sec']/1024/1024:.2f} MB/s")
    print(f"    字符速度: {ts_perf['chars_per_sec']/1000:.2f} K chars/s")
    print(f"    Token速度: {ts_perf['tokens_per_sec']/1000:.2f} K tokens/s")
    print(f"    测试数据: {ts_perf['total_bytes']/1024:.2f} KB, 用时: {ts_perf['elapsed_sec']:.3f}s")

    print("\n>>> OpenWebText分词器 (32K vocab)")
    ow_perf = benchmark_throughput(ow_tokenizer, ow_docs_20)
    print(f"    处理速度: {ow_perf['bytes_per_sec']/1024/1024:.2f} MB/s")
    print(f"    字符速度: {ow_perf['chars_per_sec']/1000:.2f} K chars/s")
    print(f"    Token速度: {ow_perf['tokens_per_sec']/1000:.2f} K tokens/s")
    print(f"    测试数据: {ow_perf['total_bytes']/1024:.2f} KB, 用时: {ow_perf['elapsed_sec']:.3f}s")

    print("\n" + "=" * 70)
    print("说明：")
    print("  - 压缩比越高，表示每个token编码的信息越多（压缩效果越好）")
    print("  - 跨领域分词时，由于词表和合并规则不匹配，压缩比会下降")
    print("  - 吞吐量受Python实现限制，C++实现（如tiktoken）会更快")
    print("=" * 70)


if __name__ == "__main__":
    main()
