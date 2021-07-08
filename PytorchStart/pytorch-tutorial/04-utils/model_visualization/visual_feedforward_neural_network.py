import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchviz import make_dot

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # self.fc = nn.Linear(input_size, num_classes)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # out = self.fc(x)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)
x = torch.rand(1, 28*28)
y = model(x)

# 这三种方式都可以
g = make_dot(y)
# g=make_dot(y, params=dict(model.named_parameters()))
#g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))


# 这两种方法都可以
# g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
g.render('visual_model.pdf', view=False)  # 会自动保存为一个 visual_model.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开