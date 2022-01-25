######导入需要的模块######
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR=0.01
BATCH_SIZE=32
EPOCH=12

######生成数据######
x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))
#print(x)
#print(y)

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True)

######构建神经网络######
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

######使用多种优化器######
net_SGD=Net()
net_Momentum=Net()
net_RMSProp=Net()
net_Adam=Net()

nets=[net_SGD,net_Momentum,net_RMSProp,net_Adam]

opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.9)
opt_RMSProp=torch.optim.RMSprop(net_RMSProp.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SGD,opt_Momentum,opt_RMSProp,opt_Adam]

######训练模型######
loss_func=torch.nn.MSELoss()
loss_his=[[],[],[],[]]
for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(loader):
        for net,opt,l_his in zip(nets,optimizers,loss_his):
            output=net(batch_x)
            loss=loss_func(output,batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())
labels=['SGD','Momentum','RMSprop','Adam']

######可视化结果######
for i,l_his in enumerate(loss_his):
    plt.plot(l_his,label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()
