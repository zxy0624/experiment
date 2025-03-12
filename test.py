import torch
import torch.nn as nn

# 1. 定义交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 2. 创建一个形状为 (3, 5) 的张量作为模型的输出（即 logits）
input = torch.randn(3, 5, requires_grad=True)
print(input)
# 3. 生成目标张量，形状为 (3, )，每个元素取值在 [0, 4]（对应 5 个类别）
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
# 4. 计算交叉熵损失
output = loss(input, target)

# 5. 对损失做反向传播，计算 input 的梯度
output.backward()
print(output)