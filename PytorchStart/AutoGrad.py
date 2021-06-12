import torch
from torch.autograd import Variable


x = Variable(torch.Tensor([[3.0, 3.0], [3.0, 3.0]]), requires_grad=True)
# x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2

print(x)
print(x.data)
print(x.grad)
# print(x.creator)

print("\n\n")
print(y)
print(y.data)
print(y.grad)
# print(y.creator)



# y 是作为一个操作的结果创建的，因此y有一个creator
z = y * y
print('z', z)

print("\n\n")
# out = z.sum()
out = z.mean()
print('out', out)

print("\n\n")
# 现在我们来使用反向传播
out.backward()

# out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
# 在此处输出 d(out)/dx
print(x.grad)


print('\n', '*'*50)
'''
现在我们来看一个雅可比向量积（vector-Jacobian product）的例子:
'''
x = torch.randn(2, requires_grad=True)
# x = torch.tensor([6.0, 2.0], requires_grad=True)
print('x:', x)
print('torch.norm(x):', torch.norm(x))
print('x.norm():', x.norm())

print('')
norm = torch.pow(x, 2)
print('norm1:', norm)
norm = norm.sum()
print('norm2:', norm)
print('norm3:', torch.sqrt(norm))

norm = x*x
print('[2]norm:', norm)


print('\n')
x = torch.randn(2, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

# 在这种情况下，y 不再是标量。torch.autograd 不能直接计算完整的雅可比矩阵，
# 但是如果我们只想要雅可比向量积，只需将这个向量作为参数传给 backward：
print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# 也可以通过将代码块包装在 with torch.no_grad(): 中，
# 来阻止autograd跟踪 .requires_grad=True 的张量的历史记录。
x = torch.randn(2, requires_grad=True)
print(x)
print(x+2)

# with torch.no_grad():
#     print((x ** 2))