"""
文本生成命令行工具

用法示例：
    uv run python cs336_basics/generate_cli.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer_vocab tokenizer/ts_vocab.pkl \
        --tokenizer_merges tokenizer/ts_merges.pkl \
        --prompt "Once upon a time" \
        --max_tokens 100 \
        --temperature 0.8 \
        --top_p 0.9
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

from cs336_basics.generate import generate
from cs336_basics.model.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def load_checkpoint_model(checkpoint_path: str, device: str, max_seq_len: int = 512):
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 运行设备
        max_seq_len: 最大序列长度（无法从检查点推断，需要手动指定）

    Returns:
        model: 加载好的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从检查点中恢复模型配置
    state_dict = checkpoint["model_state_dict"]

    # 推断词表大小（从 lm_head.weight 的形状）
    vocab_size = state_dict["lm_head.weight"].shape[0]

    # 推断模型维度（从 embedding.weight 的形状）
    # embedding.weight: [vocab_size, d_model]
    d_model = state_dict["embedding.weight"].shape[1]

    # 推断层数（从 layers 中的键数量）
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("layers."):
            layer_idx = int(key.split(".")[1])
            layer_indices.add(layer_idx)
    num_layers = max(layer_indices) + 1 if layer_indices else 1

    # 推断注意力头数（从 mthA.wq.weight 的形状）
    # Linear 层权重形状: [out_features, in_features]
    # wq: [d_model, d_model]，所以 wq.weight: [d_model, d_model]
    num_heads = None
    for key in state_dict.keys():
        if "mthA.wq.weight" in key:
            # wq 输出维度是 d_model，每个 head 的维度是 d_k = d_model / num_heads
            # 从 Linear 层权重形状 [out_features, in_features] = [d_model, d_model]
            # 无法直接推断 num_heads，但可以从第一层推断
            wq_weight = state_dict[key]
            d_k = d_model // 8  # 尝试默认 8 heads
            # 实际上从权重无法推断 num_heads，需要存储在检查点中
            # 这里我们使用默认配置
            num_heads = 8
            break

    if num_heads is None:
        num_heads = 8  # 默认值

    # 推断前馈网络维度（从 ffn.w1.weight 的形状）
    # SwiGLU 的 w1: Linear(d_model, d_ff)，权重形状 [d_ff, d_model]
    d_ff = None
    for key in state_dict.keys():
        if "ffn.w1.weight" in key:
            d_ff = state_dict[key].shape[0]  # out_features = d_ff
            break

    if d_ff is None:
        d_ff = d_model * 4  # 默认值

    print(f"加载模型配置:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  max_seq_len: {max_seq_len}")

    # 初始化模型
    model = TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=torch.device(device),
    )

    # 加载状态字典
    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="使用 Transformer 语言模型生成文本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型配置
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点文件路径",
    )
    model_group.add_argument(
        "--tokenizer_vocab",
        type=str,
        required=True,
        help="分词器词表文件路径 (.pkl)",
    )
    model_group.add_argument(
        "--tokenizer_merges",
        type=str,
        required=True,
        help="分词器合并规则文件路径 (.pkl)",
    )
    model_group.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<|endoftext|>"],
        help="特殊 token 列表",
    )

    # 生成配置
    gen_group = parser.add_argument_group("生成配置")
    gen_group.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="输入提示文本",
    )
    gen_group.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="最大生成 token 数量",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="温度参数，控制随机性 (0, inf)",
    )
    gen_group.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p 采样阈值 (0, 1]，None 表示不使用",
    )
    gen_group.add_argument(
        "--end_token",
        type=str,
        default="<|endoftext|>",
        help="结束 token，生成到该 token 时停止",
    )

    # 其他配置
    other_group = parser.add_argument_group("其他配置")
    other_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备",
    )
    other_group.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="模型最大序列长度（无法从检查点自动推断）",
    )
    other_group.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式，可以连续输入 prompt",
    )

    args = parser.parse_args()

    # 加载分词器
    print(f"加载分词器...")
    tokenizer = Tokenizer.from_files(
        args.tokenizer_vocab,
        args.tokenizer_merges,
        args.special_tokens,
    )
    print(f"词表大小: {len(tokenizer.vocab)}")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    device = torch.device(args.device)
    model = load_checkpoint_model(args.checkpoint, args.device, args.max_seq_len)
    model = model.to(device)
    print("模型加载完成！\n")

    # 打印生成配置
    print("生成配置:")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  end_token: {args.end_token}")
    print()

    if args.interactive:
        # 交互式模式
        print("进入交互式模式，输入 'quit' 或 'exit' 退出\n")
        while True:
            try:
                prompt = input("Prompt: ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                if not prompt:
                    continue

                print("\n" + "=" * 50)
                print("生成中...")
                print("=" * 50)

                generated_text = generate(
                    model=model,
                    prompt=prompt,
                    tokenizer=tokenizer,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    end_token=args.end_token,
                    device=args.device,
                )

                # 只显示生成的部分
                completion = generated_text[len(prompt) :]
                print(f"\n{prompt}{completion}")
                print("=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n\n退出交互式模式")
                break
            except Exception as e:
                print(f"\n错误: {e}\n")
    else:
        # 单次生成模式
        print("=" * 50)
        print(f"Prompt: {args.prompt}")
        print("=" * 50)
        print("\n生成中...\n")

        generated_text = generate(
            model=model,
            prompt=args.prompt,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            end_token=args.end_token,
            device=args.device,
        )

        # 只显示生成的部分
        completion = generated_text[len(args.prompt) :]
        print(f"{args.prompt}{completion}")

        print("\n" + "=" * 50)
        print("生成完成！")


if __name__ == "__main__":
    main()
