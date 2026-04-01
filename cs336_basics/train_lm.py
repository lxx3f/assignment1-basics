"""
Transformer 语言模型训练脚本

整合模型、优化器、分词器等组件，实现完整的训练流程。
"""

import argparse
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint

# 尝试导入 wandb，如果失败则不使用
try:
    import swanlab as wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None
from cs336_basics.loss import cross_entropy
from cs336_basics.model.modules import TransformerLM
from cs336_basics.optimizer import AdamW, clip_gradient_norm, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 数据配置
    train_data_path: str
    val_data_path: Optional[str] = None
    tokenizer_vocab_path: Optional[str] = None
    tokenizer_merges_path: Optional[str] = None

    # 分词器训练配置（如果需要训练新的分词器）
    train_tokenizer: bool = False
    vocab_size: int = 10000
    special_tokens: tuple = ("<|endoftext|>",)

    # 模型配置
    max_seq_len: int = 512
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0

    # 训练配置
    batch_size: int = 32
    total_iterations: int = 10000
    learning_rate: float = 3e-4
    min_learning_rate: float = 0.0
    warmup_iterations: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # 日志和检查点配置
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    log_interval: int = 10
    eval_interval: int = 100
    eval_iterations: int = 10

    # 断点续训配置
    resume: bool = False  # 是否自动从最新检查点恢复
    checkpoint_path: Optional[str] = None  # 指定特定检查点路径

    # 其他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    dtype: str = "float32"


def set_seed(seed: int):
    """设置随机种子，保证可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串

    Args:
        seconds: 秒数

    Returns:
        str: 格式化后的时间字符串，如 "1h 23m 45s"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def prepare_tokenizer(config: TrainingConfig) -> Tokenizer:
    """
    准备分词器：加载已有分词器或训练新的分词器

    Args:
        config: 训练配置

    Returns:
        Tokenizer: 分词器实例
    """
    if config.train_tokenizer or (
        config.tokenizer_vocab_path is None or config.tokenizer_merges_path is None
    ):
        print(f"训练新的 BPE 分词器，词表大小: {config.vocab_size}")
        vocab, merges = train_bpe(
            input_path=config.train_data_path,
            vocab_size=config.vocab_size,
            special_tokens=list(config.special_tokens),
        )
        # 保存分词器
        tokenizer_dir = Path(config.checkpoint_dir) / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = tokenizer_dir / "vocab.pkl"
        merges_path = tokenizer_dir / "merges.pkl"

        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)

        print(f"分词器已保存到: {tokenizer_dir}")
        return Tokenizer(vocab, merges, list(config.special_tokens))
    else:
        print(f"加载已有分词器: {config.tokenizer_vocab_path}")
        return Tokenizer.from_files(
            str(config.tokenizer_vocab_path),
            str(config.tokenizer_merges_path),
            list(config.special_tokens),
        )


def prepare_dataset(
    text_path: str, tokenizer: Tokenizer, output_path: str, dtype=np.int32
) -> np.ndarray:
    """
    将文本文件编码为 token ID 并保存为 numpy 数组

    Args:
        text_path: 文本文件路径
        tokenizer: 分词器
        output_path: 输出文件路径
        dtype: 数组数据类型

    Returns:
        np.ndarray: token ID 数组
    """
    print(f"编码数据集: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    token_array = np.array(token_ids, dtype=dtype)

    np.save(output_path, token_array)
    print(f"数据集已保存到: {output_path}, 共 {len(token_array)} 个 tokens")

    return token_array


def load_or_prepare_dataset(
    text_path: str, tokenizer: Tokenizer, cache_dir: str
) -> np.ndarray:
    """
    加载或准备数据集，使用缓存避免重复编码

    Args:
        text_path: 文本文件路径
        tokenizer: 分词器
        cache_dir: 缓存目录

    Returns:
        np.ndarray: token ID 数组
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 使用文本文件名的 hash 作为缓存文件名
    import hashlib

    text_hash = hashlib.md5(open(text_path, "rb").read()).hexdigest()[:8]
    base_name = Path(text_path).stem
    cache_file = cache_path / f"{base_name}_{text_hash}.npy"

    if cache_file.exists():
        print(f"加载缓存数据集: {cache_file}")
        return np.load(cache_file)

    return prepare_dataset(text_path, tokenizer, str(cache_file))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: np.ndarray,
    config: TrainingConfig,
) -> float:
    """
    在验证集上评估模型

    Args:
        model: 模型
        dataset: 验证数据集
        config: 训练配置

    Returns:
        float: 平均损失
    """
    model.eval()
    total_loss = 0.0
    device = torch.device(config.device)

    for _ in range(config.eval_iterations):
        x, y = get_batch(
            dataset, config.batch_size, config.max_seq_len, str(device)
        )
        logits = model(x)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        total_loss += loss.item()

    return total_loss / config.eval_iterations


def train(config: TrainingConfig):
    """
    主训练函数

    Args:
        config: 训练配置
    """
    # 设置随机种子
    set_seed(config.seed)

    # 创建检查点目录
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 设置 wandb（可选）
    if HAS_WANDB:
        wandb.init(
            project="transformer-lm",
            name=f"lm_{config.d_model}d_{config.num_layers}l",
            config=asdict(config),
        )
    else:
        print("警告: wandb 未安装，日志记录将被禁用")
        print("安装命令: pip install wandb")

    # 准备分词器
    tokenizer = prepare_tokenizer(config)
    actual_vocab_size = len(tokenizer.vocab)
    print(f"实际词表大小: {actual_vocab_size}")

    # 准备数据集
    print("准备训练数据...")
    train_data = load_or_prepare_dataset(
        config.train_data_path, tokenizer, str(checkpoint_dir / "cache")
    )

    val_data = None
    if config.val_data_path:
        print("准备验证数据...")
        val_data = load_or_prepare_dataset(
            config.val_data_path, tokenizer, str(checkpoint_dir / "cache")
        )

    # 确定数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.dtype, torch.float32)

    # 初始化模型
    print("初始化模型...")
    device = torch.device(config.device)
    model = TransformerLM(
        vocab_size=actual_vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
        dtype=torch_dtype,
    )
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 尝试从检查点恢复
    start_iteration = 0
    if config.checkpoint_path:
        # 使用指定的检查点
        checkpoint_to_load = Path(config.checkpoint_path)
        if checkpoint_to_load.exists():
            print(f"从指定检查点恢复: {checkpoint_to_load}")
            start_iteration = load_checkpoint(checkpoint_to_load, model, optimizer)
            print(f"从迭代 {start_iteration} 继续训练")
        else:
            print(f"警告: 指定的检查点不存在: {checkpoint_to_load}，从头开始训练")
    elif config.resume:
        # 自动查找最新检查点
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"从最新检查点恢复: {latest_checkpoint}")
            start_iteration = load_checkpoint(latest_checkpoint, model, optimizer)
            print(f"从迭代 {start_iteration} 继续训练")
        else:
            print("未找到检查点，从头开始训练")

    # 训练循环
    remaining_iterations = config.total_iterations - start_iteration
    print(f"开始训练，总迭代次数: {config.total_iterations}，剩余迭代次数: {remaining_iterations}")
    model.train()

    # 记录训练开始时间（用于计算总运行时间和预估剩余时间）
    train_start_time = time.time()
    iteration_start_time = train_start_time
    best_val_loss = float("inf")

    for iteration in range(start_iteration, config.total_iterations):
        # 学习率调度
        lr = get_lr_cosine_schedule(
            iteration,
            config.learning_rate,
            config.min_learning_rate,
            config.warmup_iterations,
            config.total_iterations,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 采样批次
        x, y = get_batch(
            train_data, config.batch_size, config.max_seq_len, str(device)
        )

        # 前向传播
        logits = model(x)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        clip_gradient_norm(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # 记录日志
        if iteration % config.log_interval == 0:
            current_time = time.time()
            iteration_time = current_time - iteration_start_time
            total_elapsed = current_time - train_start_time

            # 计算速度
            tokens_per_sec = (
                config.batch_size * config.max_seq_len * config.log_interval
            ) / iteration_time

            # 计算已完成的进度和预计剩余时间
            progress = (iteration - start_iteration) / remaining_iterations if remaining_iterations > 0 else 1.0
            if progress > 0:
                estimated_total_time = total_elapsed / progress
                remaining_time = estimated_total_time - total_elapsed
            else:
                remaining_time = 0

            print(
                f"Iter {iteration:6d}/{config.total_iterations} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr:.6f} | "
                f"Tokens/s: {tokens_per_sec:.0f} | "
                f"已运行: {format_time(total_elapsed)} | "
                f"预计剩余: {format_time(remaining_time)}"
            )
            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/elapsed_time": total_elapsed,
                    "train/remaining_time": remaining_time,
                }, step=iteration)
            iteration_start_time = current_time

        # 评估
        if (
            config.eval_interval > 0
            and iteration % config.eval_interval == 0
            and val_data is not None
        ):
            val_loss = evaluate(model, val_data, config)
            print(f"Iter {iteration:6d} | Val Loss: {val_loss:.4f}")
            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    "val/loss": val_loss,
                }, step=iteration)
            model.train()

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model, optimizer, iteration, best_path)
                print(f"保存最佳模型 (val_loss={val_loss:.4f}) 到 {best_path}")

        # 保存检查点
        if (
            config.checkpoint_interval > 0
            and iteration % config.checkpoint_interval == 0
            and iteration > 0
        ):
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration:06d}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"保存检查点到: {checkpoint_path}")

    # 保存最终模型
    final_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(model, optimizer, config.total_iterations, final_path)
    print(f"训练完成！最终模型保存到: {final_path}")

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="训练 Transformer 语言模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 默认工作目录：assignment1-basics/cs336_basics
    # 数据参数
    data_group = parser.add_argument_group("数据配置")
    data_group.add_argument(
        "--train_data_path", type=str, default="../data/TinyStories-train.txt", help="训练数据文件路径"
    )
    data_group.add_argument(
        "--val_data_path", type=str, default=None, help="验证数据文件路径"
    )
    data_group.add_argument(
        "--tokenizer_vocab_path", type=str, default="../tokenizer/ts_vocab.pkl", help="分词器词表文件路径"
    )
    data_group.add_argument(
        "--tokenizer_merges_path", type=str, default="../tokenizer/ts_merges.pkl", help="分词器合并规则文件路径"
    )
    data_group.add_argument(
        "--train_tokenizer",
        action="store_true",
        help="是否训练新的分词器",
    )

    # 分词器参数
    tokenizer_group = parser.add_argument_group("分词器配置")
    tokenizer_group.add_argument(
        "--vocab_size", type=int, default=10000, help="词表大小"
    )
    tokenizer_group.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<|endoftext|>"],
        help="特殊 token 列表",
    )

    # 模型参数
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--max_seq_len", type=int, default=512, help="最大序列长度"
    )
    model_group.add_argument("--d_model", type=int, default=512, help="模型维度")
    model_group.add_argument("--num_layers", type=int, default=8, help="层数")
    model_group.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    model_group.add_argument("--d_ff", type=int, default=2048, help="前馈网络维度")
    model_group.add_argument(
        "--rope_theta", type=float, default=10000.0, help="RoPE theta 参数"
    )

    # 训练参数
    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument("--batch_size", type=int, default=32, help="批次大小")
    train_group.add_argument(
        "--total_iterations", type=int, default=10000, help="总训练迭代次数"
    )
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    train_group.add_argument(
        "--min_learning_rate", type=float, default=0.0, help="最小学习率"
    )
    train_group.add_argument(
        "--warmup_iterations", type=int, default=1000, help="预热迭代次数"
    )
    train_group.add_argument(
        "--weight_decay", type=float, default=0.01, help="权重衰减"
    )
    train_group.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="最大梯度范数"
    )

    # 日志和检查点参数
    log_group = parser.add_argument_group("日志和检查点配置")
    log_group.add_argument(
        "--checkpoint_dir", type=str, default="../checkpoints", help="检查点保存目录"
    )
    log_group.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="保存检查点的间隔迭代次数",
    )
    log_group.add_argument(
        "--log_interval", type=int, default=10, help="打印日志的间隔迭代次数"
    )
    log_group.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="评估的间隔迭代次数（0 表示不评估）",
    )
    log_group.add_argument(
        "--eval_iterations", type=int, default=10, help="每次评估的迭代次数"
    )
    log_group.add_argument(
        "--resume",
        action="store_true",
        help="是否从最新检查点自动恢复训练",
    )
    log_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="指定特定的检查点文件路径（优先级高于 --resume）",
    )

    # 其他参数
    other_group = parser.add_argument_group("其他配置")
    other_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    other_group.add_argument("--seed", type=int, default=42, help="随机种子")
    other_group.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型",
    )

    args = parser.parse_args()

    # 创建配置
    config = TrainingConfig(**vars(args))

    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
