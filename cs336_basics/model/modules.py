import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear layer
    
    无bias
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        初始化Linear层
        
        Args:
            in_features: 输入的维度
            out_features: 输出的维度
            device: 参数存放的位置
            dtype: 参数的数据类型
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # 随机初始化参数
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self.weight.transpose(0, 1)

    
class Embedding(nn.Module):
    """
    嵌入层。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        初始化嵌入层。

        Args:
            num_embeddings: 词表的大小。
            embedding_dim: 嵌入向量的维度。
            device: 参数存放的位置。
            dtype: 参数的数据类型。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # 随机初始化参数
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        计算嵌入向量。

        Args:
            token_ids (torch.Tensor): 输入的token id。
        Returns:
            torch.Tensor: 嵌入向量。
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    RMSNorm层。
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        初始化RMSNorm层。

        Args:
            d_model: 模型隐藏层的维度。
            eps: 正则化项。防止分母为零。
            device: 参数存放的位置。
            dtype: 参数的数据类型。
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        # 初始化参数：全1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算RMSNorm。

        Args:
            x (torch.Tensor): 输入。
        Returns:
            torch.Tensor: 输出。
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x * rms * self.weight
        return result.to(in_dtype)


def SiLU(x: torch.Tensor) -> torch.Tensor:
    """
    计算SiLU。

    Args:
        x (torch.Tensor): 输入。
    Returns:
        torch.Tensor: 输出。
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """
    SwiGLU层。
    无bias

    计算公式:
    y = (SiLU(x*W1) \cdot (x*W3))*W2
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        初始化SwiGLU层。

        Args:
            d_model: 模型隐藏层的维度。
            d_ff: FFN的维度。
            device: 参数存放的位置。
            dtype: 参数的数据类型。
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff =  d_ff - d_ff%64
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算SwiGLU。
        计算公式:
        y = (SiLU(x*W1) \cdot (x*W3))*W2

        Args:
            x (torch.Tensor): 输入。
        Returns:
            torch.Tensor: 输出。
        """
        gate = self.w1(x)
        gate = SiLU(gate)
        feat = self.w3(x)
        return self.w2(gate * feat)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化。

        Args:
            theta (float): 基准频率 (通常为 10000)。
            d_k (int): 可以是 d_model 或 head_dim。
            max_seq_len (int): 最大序列长度。
            device (torch.device | None): 预计算的参数存放的位置
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率 θ_i = theta ^ (-2i/d_k), i = 0, 2, ..., d_k//2 -1
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # 位置索引
        positions = torch.arange(0, max_seq_len, 1, device=device)
        # 外积得到计算矩阵 [max_seq_len, d_k//2]
        pos_freqs = torch.outer(positions, freqs)
        # 重复一次，得到 [max_seq_len, d_k]
        # 注意这里的复制方式: 假设原本的一个列是[d1, d2, d3, ..., dn], 那么重复一次之后就是[d1, d1, d2, d2, ..., dn, dn]
        # 而不是[d1, d2,..., dn, d1, d2, ..., dn]
        pos_freqs = torch.repeat_interleave(pos_freqs, 2, dim=-1)
        # 预计算
        self.register_buffer("cos", torch.cos(pos_freqs), persistent=False)  # [max_seq_len, d_k]
        self.register_buffer("sin", torch.sin(pos_freqs), persistent=False)  # [max_seq_len, d_k]


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向计算

        Args:
            x: [..., seq_len, d_k] 输入张量
            token_positions: [..., seq_len] token位置索引

        Returns:
            输出矩阵
        """
        # 获取对应位置的cos和sin
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # 维度对齐
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        # 切分成偶数和奇数部分
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rot = torch.stack([-x2, x1],dim=-1).flatten(-2)
        x = x * cos + x_rot * sin
        return x

def apply_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    对输入进行softmax

    Args:
        in_features: 输入张量
        dim: softmax的维度
    """
    # 获取最大值
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    # 处理全 -inf 的情况
    max_val = max_val.masked_fill(torch.isinf(max_val) & (max_val < 0), 0.0)
    # 平移
    x_shifted = x - max_val
    x = torch.exp(x_shifted)
    return x / torch.sum(x, dim=dim, keepdim=True)    


def scaled_dot_product_attention(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
    '''
    缩放点积注意力

    Args:
        q: query张量, [..., seq_len, head_dim]
        k: key张量
        v: value张量
        mask: 掩码张量, [..., seq_len], True表示保留，False表示遮挡
    Returns:
        torch.Tensor: 缩放点积注意力结果    
    '''
    q_k_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        # mask维度对齐
        while mask.ndim < q_k_score.ndim:
            mask = mask.unsqueeze(0)
        q_k_score = q_k_score.masked_fill(mask == False, float('-inf'))
    q_k_attention = apply_softmax(q_k_score, dim=-1)
    return torch.matmul(q_k_attention, v)


class MultiHeadAttention(nn.Module):
    """
    多头注意力。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        position_embedding: nn.Module = RotaryPositionalEmbedding,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        use_position_embedding: bool = True,
        use_causal_mask: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        初始化多头注意力。

        Args:
            d_model: 模型隐藏层的维度。
            num_heads: 头的数量。
            device: 模型参数存放位置。
            dtype: 参数的数据类型。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0 , "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.position_embedding = position_embedding
        self.use_position_embedding = use_position_embedding
        if use_position_embedding:
            self.position_embedding = position_embedding(theta=theta, 
                                                        d_k=self.d_k, 
                                                        max_seq_len=max_seq_len, 
                                                        device=device)
        self.use_causal_mask = use_causal_mask
        # 注：mask的True表示保留，False表示遮挡
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len))
                .bool()
                .unsqueeze(0).unsqueeze(0),
            persistent=False
        )
        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向计算。

        Args:
            x: [batch_size, seq_len, d_model] 输入张量。
            token_positions: [batch_size, seq_len] 输入的 token 位置。
        Returns:
            torch.Tensor: 输出张量。
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, "d_model must be equal to self.d_model"
        # 投影, 调整形状为[batch_size, seq_len, num_heads, d_k]
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if self.use_position_embedding:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
            q = self.position_embedding(q, token_positions)
            k = self.position_embedding(k, token_positions)
        # 计算缩放点积注意力，得到[batch_size, num_heads, seq_len, d_k]
        attention_value = scaled_dot_product_attention(
            q, k, v,
            mask=self.causal_mask[:, :, :seq_len, :seq_len] if self.use_causal_mask else None
        )
        # 将多头的输出拼接起来，调整形状为[batch_size, seq_len, d_model]
        attention_value = attention_value.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.wo(attention_value)

        
class TransformerBlock(nn.Module):
    """
    Transformer 块。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        position_embedding: nn.Module = RotaryPositionalEmbedding,
        use_position_embedding: bool = True,
        use_causal_mask: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ) -> None:
        """
        初始化 Transformer 块。

        Args:
            d_model: 模型隐藏层的维度。
            num_heads: 头的数量。
            d_ff: 前向传播隐藏层的维度。

            device: 模型参数存放位置。
            dtype: 模型参数的数据类型。
        """
        super().__init__()
        self.mthA = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            position_embedding=position_embedding,
            use_position_embedding=use_position_embedding,
            use_causal_mask=use_causal_mask,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rmsn1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.rmsn2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向计算。

        Args:
            x: [batch_size, seq_len, d_model] 输入张量。
            token_positions: [batch_size, seq_len] 输入的 token 位置。
        Returns:
            torch.Tensor: 输出张量。
        """
        x = x + self.mthA(self.rmsn1(x), token_positions)
        return x + self.ffn(self.rmsn2(x))


class TransformerLM(nn.Module):
    """
    Transformer 语言模型。
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ):
        """
        初始化 Transformer 语言模型。

        Args:
            vocab_size: 词汇表的大小。
            max_seq_len: 模型输入的序列最大长度。
            d_model: 模型隐藏层的维度。
            num_layers: 模型块的数量。
            num_heads: 头的数量。
            d_ff: 前向传播隐藏层的维度。
            rope_theta: ROPE 的缩放参数。
            device: 模型参数存放位置。
            dtype: 模型参数的数据类型。
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype
                ) for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向计算。

        Args:
            x: [batch_size, seq_len] 输入张量。
            token_positions: [batch_size, seq_len] 输入的 token 位置。
        Returns:
            torch.Tensor: 输出张量。
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x






