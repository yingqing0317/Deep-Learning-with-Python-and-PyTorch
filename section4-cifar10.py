######加载数据######
# 导入库及下载数据
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='D:\\Audrey\\zju\\science\\pystudy\\pytorch_study\\data\\CIFAR-10',
                                        train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='D:\\Audrey\\zju\\science\\pystudy\\pytorch_study\\data\\CIFAR-10',
                                       train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#####构建网络######
# 根据图6-1构建网络
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36 * 6 * 6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x


net = CNNNet()
net = net.to(device)

# 初始化参数
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight)
        # nn.init.xavier_normal(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        # nn.init.constant_(m.bias,0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)

######训练模型######
# 选择优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#训练模型
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter(log_dir='D:\\Audrey\\zju\\science\\pystudy\\pytorch_study\\data\\CIFAR-10',comment='feature map')

img_grid=vutils.make_grid(x,normalize=True,scale_each=True,nrow=2)
net.eval()
for name,layer in net.named_modules.item():
    x=x.view(x.size(0),-1) if "fc" in name else x
    print(x.size())

    x=layer(x)
    print(f'{name}')

    if 'layer' in name or 'conv' in name:
        x1=x.transpose(0,1)
        img_grid=vutils.make_grid(x1,normalize=True,scale_each=True,nrow=4)
        writer.add_image(f'{name}_feature_maps',img_grid,global_step=0)