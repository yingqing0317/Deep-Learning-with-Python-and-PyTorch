###### 4.2 util.data简介 ######
'''
import torch
from torch.utils import data
import numpy as np

class TestDataset(data.Dataset):
    def __init__(self):
        self.Data=np.asarray([[1,2],[3,4],[2,1],[3,4],[4,5]])
        self.Label=np.asarray([0,1,0,1,2])
    def __getitem__(self, index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    def __len__(self):
        return len(self.Data)

Test=TestDataset()
print(Test[2])
print(Test.__len__())

#以下部分仅演示dataloader的用法
#test_loader=data.DataLoader(torch.randn(1000,2),batch_size=5,shuffle=False)
#for i in test_loader:
#    print(i)
#    break

#此部分是运用我们之前加载的数据集
test_loader=data.DataLoader(Test,batch_size=2,shuffle=False,num_workers=0)
for i,traindata in enumerate(test_loader):
    print('i:',i)
    Data,Label=traindata
    print('data:',Data)
    print('Label:',Label)
'''
######4.3 torchvision######
'''
from torchvision import transforms,utils
from torchvision import datasets
import torch
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image

my_trans=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_data=datasets.ImageFolder('./data/torchvision_data',transform=my_trans)
# 得自己创建一个名为torchvision_data的文件夹，并下载好图片
train_loader=data.DataLoader(train_data,batch_size=8,shuffle=True,)

for i_batch,img in enumerate(train_loader):
    if i_batch==0:
        print(img[1])
        fig=plt.figure()
        grid=utils.make_grid(img[0],4)
        plt.imshow(grid.numpy().transpose((1,2,0)))
        plt.show()
        utils.save_image(grid,'test01.png')
    break

Image.open('test01.png') 
'''
###### 4.4 tensorboardX ######

#导入需要的模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
import numpy as np

###### 4.4.1 用tensorboardX可视化神经网络 ######
#构建神经网络

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 10,kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)
        self.bn=nn.BatchNorm2d(20)
    def forward(self,x):
        x=F.max_pool2d(self.conv1(x),2)
        x=F.relu(x)+F.relu(-x)
        x=F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x=self.bn(x)
        x=x.view(-1,320)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,training=self.training)
        x=self.fc2(x)
        x=F.softmax(x,dim=1)
        return x

#把模型保存为graph
input=torch.rand(32,1,28,28)
model=Net()
with SummaryWriter(log_dir=r'D:\Audrey\zju\science\pystudy\pytorch_study\logs',comment='Net') as w:
    w.add_graph(model,(input,))


######4.4.3 用tensorboardX 可视化损失值######
'''
input_size=1
output_size=1
learning_rate=0.01
num_epoches=60

dtype=torch.FloatTensor
writer=SummaryWriter(log_dir=r'D:\Audrey\zju\science\pystudy\pytorch_study\logs',comment='Linear')
np.random.seed(100)
x_train=np.linspace(-1,1,100).reshape(100,1)
y_train=3*np.power(x_train,2)+2+0.2*np.random.rand(x_train.size).reshape(100,1)

model=nn.Linear(input_size,output_size)

criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(num_epoches):
    inputs=torch.from_numpy(x_train).type(dtype)
    targets=torch.from_numpy(y_train).type(dtype)

    output=model(inputs)
    loss=criterion(output,targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('训练损失值',loss,epoch)

#writer.flush()
#writer.close()
'''
