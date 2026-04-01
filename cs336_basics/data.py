import torch
import numpy as np
import numpy.typing as npt
import os
import typing

def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 Numpy 数组数据集中随机采样一个批次。
    
    参数:
        dataset: 1D Numpy 数组 (Token IDs)
        batch_size: 批大小 (B)
        context_length: 上下文长度 (m)
        device: 设备字符串 ('cpu', 'cuda', 'mps')
    """
    # 1. 确定合法的最大起点索引
    # 我们需要取长度为 context_length 的片段，且目标还要往后偏移一位
    # 所以最后一个可用的起点是 len(dataset) - context_length - 1
    n = len(dataset)
    max_idx = n - context_length - 1
    
    # 2. 随机产生 batch_size 个起始位置, 这些位置不会重合，但采样区域可能重叠
    # np.random.randint 在 [0, max_idx] 之间产生随机整数
    ix = torch.randint(0, max_idx + 1, (batch_size,))
    
    # 3. 根据索引提取输入和目标
    # x: dataset[i : i+m]
    # y: dataset[i+1 : i+m+1]
    # 我们先在 CPU 上提取数据，然后一次性转为 Tensor
    x_stack = [dataset[i : i + context_length] for i in ix]
    y_stack = [dataset[i + 1 : i + context_length + 1] for i in ix]
    
    # 4. 转换为 PyTorch 张量并移动到指定设备
    # 注意：dataset 通常是 int32 或 int64，转为 torch 后通常使用 torch.long (int64)
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()
    
    return x, y




def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    保存当前训练状态。
    """
    # 1. 构建一个包含所有必要信息的字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 2. 使用 torch.save 将字典写入目标（可以是路径或文件流）
    torch.save(checkpoint, out)

def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点恢复状态，并返回保存时的迭代次数。
    """
    # 1. 加载字典
    # 使用 map_location='cpu' 是一个好习惯，可以防止在没有 GPU 的机器上加载时报错
    checkpoint = torch.load(src, map_location='cpu')
    
    # 2. 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 恢复优化器状态（动量、步数等）
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 4. 返回保存时的迭代次数
    return checkpoint['iteration']