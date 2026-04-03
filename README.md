# CS336 Spring 2025 Assignment 1: Basics 总结报告

## 作业概述

本作业从零实现了基于 PyTorch 的 Transformer 语言模型，包括 BPE 分词器训练、完整模型架构、训练流程和文本生成工具。

## 完成内容

### 1. 神经网络基础模块 (`cs336_basics/model/modules.py`)

- **Linear**: 无偏置的线性层，使用截断正态分布初始化
- **Embedding**: 嵌入层，支持整数索引到稠密向量的映射
- **RMSNorm**: 均方根层归一化，用于替代 LayerNorm
- **SiLU & SwiGLU**: 激活函数及 SwiGLU FFN 结构
- **RotaryPositionalEmbedding (RoPE)**: 旋转位置编码，支持预计算 cos/sin 缓存
- **Scaled Dot-Product Attention**: 实现注意力计算，支持因果掩码
- **MultiHeadAttention**: 多头自注意力，支持 RoPE 和因果掩码
- **TransformerBlock**: Pre-norm 结构的 Transformer 块
- **TransformerLM**: 完整的 Transformer 语言模型

### 2. BPE 分词器 (`cs336_basics/tokenizer.py`, `train_bpe.py`)

- 实现 BPE (Byte-Pair Encoding) 分词器训练算法
- 支持特殊 token 处理 (`<|endoftext|>`)
- 包含预分词（按 GPT-2 正则表达式分割）
- 实现 `encode` 和 `decode` 方法
- 支持从文件保存/加载词表和合并规则

### 3. 训练工具

#### 优化器 (`optimizer.py`)
- **AdamW**: 实现 AdamW 优化器（带权重衰减的 Adam）
- **梯度裁剪**: 支持按 L2 范数裁剪梯度
- **余弦学习率调度**: 支持 warmup + cosine decay

#### 损失函数 (`loss.py`)
- 实现交叉熵损失函数

#### 数据加载 (`data.py`)
- 实现批量采样（支持随机采样长序列）
- 检查点保存/加载功能（模型 + 优化器状态 + 迭代次数）

### 4. 训练流程 (`train_lm.py`)

- 完整的训练脚本，支持：
  - 分词器训练或从文件加载
  - 数据集编码和缓存
  - 验证集评估
  - 定期保存检查点
  - 支持断点续训
  - 集成 SwanLab/Wandb 日志记录
  - 实时显示训练进度、loss、学习率、tokens/s 等

### 5. 文本生成 (`generate.py`, `generate_cli.py`)

- 实现自回归文本生成
- 支持温度采样 (temperature scaling)
- 支持 Top-p (nucleus) 采样
- 支持批量生成
- 命令行工具支持交互式模式

## 模型配置

训练完成的模型配置：

| 参数 | 值 |
|------|-----|
| 词表大小 | 10,000 |
| 模型维度 (d_model) | 512 |
| 层数 (num_layers) | 8 |
| 注意力头数 (num_heads) | 8 |
| FFN 维度 (d_ff) | 2048 |
| 最大序列长度 | 512 |
| 位置编码 | RoPE (θ=10000) |

## 训练结果

- 在 TinyStories 数据集上训练
- 检查点保存在 `checkpoints/final_model.pt`
- 模型可生成连贯的短篇故事

## 测试验证

所有实现通过 pytest 测试套件，包括：
- 各模块的前向传播正确性（与参考实现对比）
- 检查点保存/加载的一致性
- 分词器编码/解码的正确性

## 使用示例

```bash
# 单次生成
uv run python cs336_basics/generate_cli.py --prompt "Once upon a time" 

# 交互式
uv run python cs336_basics/generate_cli.py --interactive
```

