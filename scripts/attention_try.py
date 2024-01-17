import torch
import torch.nn.functional as F

def dot_product_attention(query, key, value):
    """
    点积注意力机制的实现
    Args:
    - query: 查询张量
    - key: 键张量
    - value: 值张量
    Returns:
    - attention_output: 注意力输出
    """
    # 计算注意力分数
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    # 缩放
    attention_scores = attention_scores / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))

    # Softmax得到注意力权重
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 加权求和得到输出
    attention_output = torch.matmul(attention_weights, value)

    return attention_output

# 示例使用
query = torch.rand((1, 10, 32))  # 10个查询，每个查询有32维
key = torch.rand((1, 10, 32))    # 10个键，每个键有32维
value = torch.rand((1, 10, 64))  # 10个值，每个值有64维

attention_output = dot_product_attention(query, key, value)
print(attention_output.shape)  # 输出的形状, [1, 10, 64]
