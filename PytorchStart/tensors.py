from __future__ import print_function
import torch

x = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
# 此处在notebook中输出x的值来查看具体的x内容
print(x)
print(x.size())


### NOTE: torch.Size 事实上是一个tuple, 所以其支持相关的操作*
y = torch.rand(5, 3)


### 此处 将两个同形矩阵相加有两种语法结构
# 语法一
print(x + y)
# 语法二
print(torch.add(x, y))

### 另外输出tensor也有两种写法
# 语法一
result = torch.Tensor(5, 3)
# 语法二
torch.add(x, y, out=result)

# 将y与x相加
y.add_(x)

# 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
# 例如：x.copy_(y), x.t_(), 这俩都会改变x的值。

#另外python中的切片操作也是支持的

x[:, 1]  # 这一操作会输出x矩阵的第二列的所有值