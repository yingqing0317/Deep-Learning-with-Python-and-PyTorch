# %%
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from torch import nn

from torch import optim

# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data = datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = transform,
    download = True
)

test_data = datasets.CIFAR10(
    root = "./data", 
    train = False,
    transform = transform, 
    download  = True
)

batch_size = 4
trainloader = DataLoader(train_data, batch_size, shuffle = False, num_workers = 0)
testloader = DataLoader(test_data, batch_size, shuffle = False, num_workers = 0)

# %%
dataiter = iter(testloader)
image, label = dataiter.next()
print(f"imput:[B, C, H, W] {image.size()}")
print(f"output: {label.size()}")

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_maxpool_fc = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(start_dim = 1),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        return self.conv_relu_maxpool_fc(x)

net = Net()
print(net)
i = 0
for para in net.parameters():
    print(f"para{i}: {para.size()}")
    i = i + 1

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

net = net.to(device)

# %%
print(f"trainloder共有{len(trainloader.dataset)}组数据")
print(f"trainloder的每个batch有{len(trainloader)}组数据")
dataiter = iter(trainloader)
input, output = dataiter.next()
print(f"input的size为：{input.size()}, output的size为：{output.size()}")

# %%
def train(dataloder, model, optimizer, loss_fn):
    size = len(dataloder.dataset)
    sumloss = 0
    for batch, (X, y) in enumerate(dataloder):
        X = X.to(device)
        y = y.to(device)
        # X, y = X.to(device), y.to(device)
        pred = model.forward(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sumloss += loss.item()
        if batch % 2000 == 1999:
            current = (batch + 1) * len(X)
            print(f"loss: {sumloss / 2000:>7f}  [{current:>5d}/{size:>5d}]")
            sumloss = 0
        
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct): 0.1f}%, Avg loss: {test_loss:.8f}")
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
epoch = 2
for i in range(epoch):
    print(f"Epoch{i + 1}\n--------------------------------")
    train(trainloader, net, optimizer, loss_fn)
    test(testloader, net, loss_fn)
print("Done")

# %%
print(len(testloader))
print(len(testloader.dataset))

# %%
net.eval()

# %%
print(net)

# %%
a = torch.ones(3, 4, 5)
print(a)
print(len(a))

# %%
for X,y in testloader:
    print(X.size(), y.size())
    break

# %%
a = torch.tensor([[1, 2, 3, 4],[3, 4, 5, 6]])

# %%
a.size()

# %%
a.max(1)

# %%
a = torch.tensor(1)
b = torch.tensor(1)
a.equal(b)


