import torch

#2.4.1 Tensor概述
x=torch.tensor([1,2])
y=torch.tensor([3,4])
z=x.add(y)
print(x)
print(y)
print(z)
x.add_(y)
print(x)

#2.4.2 创建Tensor
'''
a=torch.Tensor([1,2,3,4,5,6])
b=torch.Tensor(2,3)
t=torch.Tensor([[1,2,3],[4,5,6]])
print(a)
print(b)
print(t)

print(t.size())
print(t.shape)
print(torch.Tensor(t.size()))

t1=torch.Tensor(1)
t2=torch.Tensor(1)
print("t1的值:{},t1的数据类型:{}".format(t1,t1.type()))
print("t2的值:{},t2的数据类型:{}".format(t2,t2.type()))

print(torch.eye(2,2))
print(torch.zeros(2,3))
print(torch.linspace(1,10,4))
print(torch.rand(2,3))
print(torch.randn(2,3))
print(torch.zeros_like(torch.rand(2,3)))
'''
#2.4.3 修改Tensor形状
'''
x=torch.randn(2,3)
print(x)
print(x.size)
print(x.dim)
print(x)
print(x.view(3,2))
y=x.view(-1)
print("展平为1维向量后y、形状、维度：{},{},{}".format(y,y.size,y.dim))
z=torch.unsqueeze(y,0)
print("展平为1维向量后z、形状、维度、元素个数：{},{},{},{}".format(z,z.size,z.dim,z.numel))
'''
#2.4.4索引操作
'''
torch.manual_seed(100)
x=torch.randn(2,3)
print("随机产生的x的值：{}".format(x))
print("x第一行：{}".format(x[0,:]))
print("x最后一列：{}".format(x[:,-1]))
#生成是否大于0的Byter张量
mask=x>0
print("大于0的值：{}".format(torch.masked_select(x,mask)))
print("非0元素的下标：{}".format(torch.nonzero(mask)))
#获取指定索引的值
index=torch.LongTensor([[0,1,1]])
print("index为：{}".format(index))
print("指定索引的值为：{}".format(torch.gather(x,0,index)))

index=torch.LongTensor([[0,1,1],[1,1,1]])
print("index为：{}".format(index))
a=torch.gather(x,1,index)
print("指定索引的值（a值）为：{}".format(a))

z=torch.zeros(2,3)
print("将a的值返回到2x3的矩阵中：{}".format(z.scatter_(1,index,a)))
'''
#2.4.5广播机制
'''
import numpy as np
A=np.arange(0,40,10).reshape(4,1)
B=np.arange(0,3)
A1=torch.from_numpy(A)
B1=torch.from_numpy(B)
C=A1+B1
print(A)
print(B)
print(A1)
print(B1)
print(C)

B2=B1.unsqueeze(0)
A2=A1.expand(4,3)
B3=B2.expand(4,3)
C1=A2+B3
print(B2)
print(A2)
print(B3)
print(C1)
'''
#2.4.6逐元素操作
'''
t=torch.randn(1,3)
print(t)
t1=torch.randn(3,1)
t2=torch.randn(1,3)
t3=torch.addcdiv(t,0.1,t1,t2)
t4=torch.sigmoid(t)
t5=torch.clamp(t,0,1)
t6=t.add_(2)
print(t)
print(t1)
print(t2)
print(t3)
print(t4)
print(t5)
print(t6)
'''
#2.4.7归并操作
'''
a=torch.linspace(0,10,6)
print(a)
a=a.view((2,3))
print(a)
b=a.sum(dim=0)
print(b)
b=a.sum(dim=0,keepdim=True)
print(b)
'''
#2.4.8比较操作
'''
x=torch.linspace(0,10,6).view(2,3)
xx=torch.linspace(0,10,6).view((2,3))
print(x)
print(xx)
print(torch.max(x))
print(torch.max(x,dim=0))
print(torch.topk(x,1,dim=0))
'''
#2.4.9矩阵操作
'''
a=torch.tensor([2,3])
b=torch.tensor([3,4])
print(a)
print(b)
print(torch.dot(a,b))
x=torch.randint(10,(2,3))
y=torch.randint(6,(3,4))
print(x)
print(y)
print(torch.mm(x,y))
x=torch.randint(10,(2,2,3))
y=torch.randint(6,(2,3,4))
print(x)
print(y)
print(torch.bmm(x,y))
'''
######pytorch######
#2.5.3 标量反向传播
'''
x=torch.Tensor([2])
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
y=torch.mul(w,b)
z=torch.add(y,b)
print("x,w,b的require_grad属性分别为：{}，{}，{}".format(x.requires_grad,w.requires_grad,b.requires_grad))
print(x)
print(w)
print(b)
print(y)
print(z)
print("y,z的require_grad属性分别为：{}，{}".format(y.requires_grad,z.requires_grad))
print("x,w,b,y,z是否为叶子节点:{},{},{},{},{}".format(x.is_leaf,w.is_leaf,b.is_leaf,y.is_leaf,z.is_leaf))
print("x,w,b,y,z的grad_fn属性:{},{},{},{},{}".format(x.grad_fn,w.grad_fn,b.grad_fn,y.grad_fn,z.grad_fn))
z.backward()
print("x,w,b的梯度为:{},{},{}".format(x.grad,w.grad,b.grad))
#print("x,w,b,y,z的梯度为:{},{},{},{},{}".format(x.grad,w.grad,b.grad,y.grad,z.grad))
'''
#2.5.4 非标量反向传播
'''
import torch
x=torch.tensor([[2,3]],dtype=torch.float,requires_grad=True)
J=torch.zeros(2,2)
y=torch.zeros(1,2)
y[0,0]=x[0,0]**2+3*x[0,1]
y[0,1]=x[0,1]**2+2*x[0,0]
y.backward(torch.Tensor([[1,0]]),retain_graph=True)
J[0]=x.grad
x.grad=torch.zeros_like(x.grad)
y.backward(torch.Tensor([[0,1]]))
J[1]=x.grad
print(J)
'''
######2.6 使用Numpy实现机器学习(即不使用pytorch任何包或类)######
'''
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(100)
x=np.linspace(-1,1,100).reshape(100,1)
y=3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
#print(x)
#print(y)
plt.scatter(x,y)
w1=np.random.rand(1,1)
b1=np.random.rand(1,1)
print("w1初始值为：{}".format(w1))
print("b1初始值为：{}".format(b1))
#plt.show()
lr=0.001
for i in range(800):
    y_pred=np.power(x,2)*w1+b1
    loss=0.5*(y-y_pred)**2
    loss=loss.sum()
    grad_w=np.sum((y_pred-y)*np.power(x,2))
    grad_b=np.sum((y_pred-y))
    w1-=lr*grad_w
    b1 -= lr * grad_b
plt.plot(x,y_pred,'r-',label='predict')
plt.scatter(x,y,color='blue',marker='o',label='true')#散点图
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()#图例
print("w1训练结果为：{}".format(w1))
print("b1训练结果为：{}".format(b1))
plt.show()
'''
######2.7 使用Tensor及Antograd实现机器学习######
'''
from matplotlib import pyplot as plt
torch.manual_seed(100)
dtype=torch.float
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=3*x.pow(2)+2+0.2*torch.rand(x.size())
plt.scatter(x.numpy(),y.numpy())
#plt.show()
w = torch.randn(1,1,dtype=dtype,requires_grad=True)
b = torch.randn(1,1,dtype=dtype,requires_grad=True)
lr=0.001
for ii in range(800):
    y_pred=x.pow(2).mm(w)+b
    loss=0.5*(y_pred-y)**2
    loss=loss.sum()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
plt.plot(x.numpy(),y_pred.detach().numpy(),'r-',label='predict')
plt.scatter(x.numpy(),y.numpy(),color='blue',marker='o',label='true')
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
print("w训练结果为：{}".format(w))
print("b训练结果为：{}".format(b))
plt.show()
'''
######2.8 TensorFlow架构 ######
'''
import tensorflow as tf
import numpy as np
np.random.seed(100)
x = np.linspace(-1,1,100).reshape(100,1)
y = 3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
x1 = tf.placeholder(tf.float32,shape=(None,1))
y1 = tf.placeholder(tf.float32,shape=(None,1))
w = tf.Variable(tf.random_uniform([1],0,1.0))
b = tf.Variable(tf.zeros([1]))
y_pred = np.power(x,2)*w+b
loss=tf.reduce_mean(tf.square(y-y_pred))
grad_w,grad_b=tf.gradients(loss,[w,b])
learning_rate=0.01
new_w = w.assign(w-learning_rate*grad_w)
new_b = b.assign(b-learning_rate*grad_b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        loss_value,v_w,v_b = sess.run([loss,new_w,new_b],[feed_dict={x1:x,y1:y}])
        if step%200==0:
            print("损失值、权重、偏移量分别为{:.4f],{},{}".format(loss_value,v_w,v_b))
plt.figure()
plt.scatter(x,y)
plt.plot(x,v_b,v_w*x**2)
'''

