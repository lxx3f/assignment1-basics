"""
Transformer 语言模型文本生成模块

支持功能：
- 根据 prompt 生成文本补全
- 温度缩放 (temperature scaling)
- Top-p 采样 (nucleus sampling)
- 最大生成长度控制
- 自动停止于特殊 token (<|endoftext|>)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用温度缩放

    Args:
        logits: [batch_size, vocab_size] 未归一化的 logits
        temperature: 温度参数，> 0
            - temperature < 1: 更确定，更尖锐的分布
            - temperature = 1: 原始分布
            - temperature > 1: 更随机，更平坦的分布

    Returns:
        torch.Tensor: 缩放后的 logits
    """
    if temperature <= 0:
        raise ValueError(f"温度必须大于 0，当前值: {temperature}")
    return logits / temperature


def top_p_sampling(
    probs: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Top-p (nucleus) 采样

    从累积概率超过 top_p 的最小 token 集合中采样

    Args:
        probs: [vocab_size] 概率分布
        top_p: 累积概率阈值 (0, 1]
        min_tokens_to_keep: 至少保留的 token 数量

    Returns:
        torch.Tensor: 采样得到的 token ID
    """
    if top_p <= 0 or top_p > 1:
        raise ValueError(f"top_p 必须在 (0, 1] 范围内，当前值: {top_p}")

    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到累积概率超过 top_p 的位置
    # 保留所有累积概率 <= top_p 的 token，以及至少 min_tokens_to_keep 个
    sorted_indices_to_remove = cumulative_probs > top_p
    # 保留第一个超过阈值的 token（使得累积概率刚好超过 top_p）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # 确保至少保留 min_tokens_to_keep 个 token
    if sorted_indices_to_remove.shape[-1] > min_tokens_to_keep:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # 将被移除的 token 的概率设为 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs_filtered = probs.clone()
    probs_filtered[indices_to_remove] = 0

    # 重新归一化
    probs_filtered = probs_filtered / probs_filtered.sum()

    # 采样
    next_token = torch.multinomial(probs_filtered, num_samples=1)

    return next_token


def generate(
    model: torch.nn.Module,
    prompt: str,
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    end_token: Optional[str] = "<|endoftext|>",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """
    使用语言模型生成文本补全

    Args:
        model: Transformer 语言模型
        prompt: 输入提示文本
        tokenizer: 分词器（需支持 encode/decode）
        max_tokens: 最大生成 token 数量
        temperature: 温度参数，控制随机性
            - 1.0: 原始分布
            - < 1.0: 更确定
            - > 1.0: 更随机
        top_p: Top-p 采样阈值 (0, 1]，None 表示不使用
        end_token: 结束 token，生成到该 token 时停止，None 表示不停止
        device: 运行设备

    Returns:
        str: 生成的完整文本（包含 prompt）
    """
    model.eval()

    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 获取结束 token 的 ID
    end_token_id = None
    if end_token is not None:
        end_token_ids = tokenizer.encode(end_token)
        # 假设特殊 token 只对应一个 ID
        if len(end_token_ids) == 1:
            end_token_id = end_token_ids[0]

    generated_ids = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            # 如果序列太长，截断到 max_seq_len
            if input_tensor.shape[1] > model.max_seq_len:
                input_tensor = input_tensor[:, -model.max_seq_len:]

            # 前向传播
            logits = model(input_tensor)  # [1, seq_len, vocab_size]

            # 获取最后一个位置的 logits
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # 应用温度缩放
            if temperature != 1.0:
                next_token_logits = apply_temperature_scaling(next_token_logits, temperature)

            # 计算概率分布
            probs = F.softmax(next_token_logits, dim=-1)

            # 采样下一个 token
            if top_p is not None and top_p < 1.0:
                next_token = top_p_sampling(probs, top_p)
            else:
                # 贪婪采样或温度采样
                if temperature == 0:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            next_token_id = next_token.item()

            # 检查是否生成结束 token
            if end_token_id is not None and next_token_id == end_token_id:
                break

            # 添加到生成的序列
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

    # 解码生成结果
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def generate_batch(
    model: torch.nn.Module,
    prompts: list[str],
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    end_token: Optional[str] = "<|endoftext|>",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[str]:
    """
    批量生成文本（所有 prompt 使用相同的生成长度）

    Args:
        model: Transformer 语言模型
        prompts: 输入提示文本列表
        tokenizer: 分词器
        max_tokens: 最大生成 token 数量
        temperature: 温度参数
        top_p: Top-p 采样阈值
        end_token: 结束 token
        device: 运行设备

    Returns:
        list[str]: 生成的文本列表
    """
    model.eval()

    # 编码所有 prompts
    batch_input_ids = [tokenizer.encode(p) for p in prompts]
    batch_size = len(prompts)

    # 获取结束 token 的 ID
    end_token_id = None
    if end_token is not None:
        end_token_ids = tokenizer.encode(end_token)
        if len(end_token_ids) == 1:
            end_token_id = end_token_ids[0]

    # 跟踪每个序列是否已完成
    finished = [False] * batch_size
    generated_ids = [ids.copy() for ids in batch_input_ids]

    with torch.no_grad():
        for step in range(max_tokens):
            # 如果所有序列都已完成，提前退出
            if all(finished):
                break

            # 构建 batch 输入（需要 padding）
            max_len = max(len(ids) for ids in generated_ids)
            input_ids_padded = []
            attention_mask = []

            for ids in generated_ids:
                padding_length = max_len - len(ids)
                padded = [0] * padding_length + ids  # 使用 0 作为 pad token
                mask = [0] * padding_length + [1] * len(ids)
                input_ids_padded.append(padded)
                attention_mask.append(mask)

            input_tensor = torch.tensor(input_ids_padded, dtype=torch.long, device=device)

            # 如果序列太长，截断到 max_seq_len
            if input_tensor.shape[1] > model.max_seq_len:
                input_tensor = input_tensor[:, -model.max_seq_len:]

            # 前向传播
            logits = model(input_tensor)  # [batch, seq_len, vocab_size]

            # 获取每个序列最后一个有效位置的 logits
            next_tokens = []
            for i in range(batch_size):
                if finished[i]:
                    next_tokens.append(torch.tensor([0], device=device))
                    continue

                seq_len = len(generated_ids[i])
                next_token_logits = logits[i, seq_len - 1, :]  # 最后一个位置的 logits

                # 应用温度缩放
                if temperature != 1.0:
                    next_token_logits = apply_temperature_scaling(next_token_logits, temperature)

                # 计算概率分布
                probs = F.softmax(next_token_logits, dim=-1)

                # 采样
                if top_p is not None and top_p < 1.0:
                    next_token = top_p_sampling(probs, top_p)
                else:
                    if temperature == 0:
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)

                next_tokens.append(next_token)

                next_token_id = next_token.item()

                # 检查是否生成结束 token
                if end_token_id is not None and next_token_id == end_token_id:
                    finished[i] = True
                else:
                    generated_ids[i].append(next_token_id)

    # 解码生成结果
    generated_texts = [tokenizer.decode(ids) for ids in generated_ids]

    return generated_texts
