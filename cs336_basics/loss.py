import torch


def cross_entropy(
        logits: torch.Tensor, 
        targets: torch.Tensor,
        ignore_index: int = -100
        ) -> torch.Tensor:
    """
    计算交叉熵损失

    Args:
        logits (torch.Tensor): [..., vocab_size] 输入的预测结果。
        targets (torch.Tensor): [...] 目标标签。
        ignore_index (int, optional): 忽略的索引。默认为 -100。
    Returns:
        torch.Tensor: 交叉熵损失。
    """
    mask = (targets != ignore_index)  # True = 有效, False = 忽略
    targets_masked = targets.masked_fill(~mask, 0)

    # 平移
    max_val, _ = torch.max(logits, dim=-1, keepdim=True)
    max_val = max_val.masked_fill(torch.isinf(max_val) & (max_val < 0), 0.0)
    shifted_logits = logits - max_val
    shifted_logits_exp = torch.exp(shifted_logits)
    target_logits = torch.gather(
        logits, dim=-1, index=targets_masked.unsqueeze(-1)
    ).squeeze(-1)
    log_sum_exp = max_val.squeeze(-1) + torch.log(torch.sum(shifted_logits_exp, dim=-1))
    loss = log_sum_exp - target_logits
    loss = loss.masked_fill(~mask, 0.0)
    num_valid = mask.sum()
    if num_valid == 0:
        return loss.sum()  # 安全：防止除以0
    return loss.sum() / num_valid



