import numpy as np

#1.1.1列表转换为ndarray

lst1=[3.14,2.17,0,1,2]
nd1=np.array(lst1)
print(nd1)
print(type(nd1))

lst2=[[3.14,2.17,0,1,2],[1,2,3,4,5]]
nd2=np.array(lst2)
print(nd2)
print(type(nd2))


#1.1.2random
'''
nd3=np.random.random([3,3])
print(nd3)
nd33=np.random.random([3,3])
print(nd33)#确实会生成俩不一样的数组
print("nd3的形状为：",nd3.shape)

np.random.seed(123)#基于随机种子实现代码的随机方法更能保证代码的可复现性
nd4=np.random.randn(2,3)
print(nd4)
np.random.shuffle(nd4)
print("随机打乱后的数据：")
print(nd4)
print(type(nd4))
'''

#1.1.3特定形状array
'''
nd5=np.zeros([3,3])
nd55=np.zeros_like(nd5)
nd6=np.ones([3,3])
nd7=np.eye(3)
nd8=np.diag([1,2,3])
print(nd5)
print(nd55)
print(nd6)
print(nd7)
print(nd8)

nd9=np.random.random([5,5])
print(nd9)
np.savetxt(X=nd9,fname='./test1.txt')#相对地址
nd10=np.loadtxt('./test1.txt')
print(nd10)#nd9完全一样
'''

#1.1.4 arange、linspace
'''
print(np.arange(10))#一个r
print(np.arange(0,10))
print(np.arange(0,4,0.5))#不包括终点（4）本身
print(np.arange(9,-1,-1))#不包括终点（-1）本身

print(np.linspace(0,1,10))#把0-1分成十个数，且包括0、1本身（即分成9等份）
print(np.linspace(0,1,11))
'''

#1.2获取元素
'''
np.random.seed(2019)
nd11=np.random.random([10])
print(nd11)
print(nd11[3])
print(nd11[3:6])#不包括终点
print(nd11[1:6:2])
print(nd11[::-2])

nd12=np.arange(25).reshape([5,5])
print(nd12)
print(nd12[1:3,1:3])
print(nd12[(nd12>3)&(nd12<10)])
print(nd12[[1,2]])
print(nd12[1:3,:])
print(nd12[:,1:3])

from numpy import random as nr
a=np.arange(1,25,dtype=float)
print("原数组：",a)#逗号分中英文
print("随机可重复抽取：",nr.choice(a,size=(3,4)))#不能在print里赋值，即以下程序是错误的
#print("随机可重复抽取：",c1=nr.choice(a,size=(3,4)))
print("随机不可重复抽取：",nr.choice(a,size=(3,4),replace=False))#replace=false不可重复抽取
print("随机但按制度概率抽取：",nr.choice(a,size=(3,4),p=a/np.sum(a)))
'''

#1.3 numpy的算术运算
#1.3.1 对应元素相乘
'''
A=np.array([[1,2],[-1,4]])
B=np.array([[2,0],[3,4]])
print(A)
print(B)
print(A*B)
print(np.multiply(A,B))

X=np.random.rand(2,3)
def softmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
print(X)
print(softmoid(X))
print(relu(X))
print(softmax(X))
'''

#1.3.2点积/内积
'''
X1=np.array([[1,2],[3,4]])
X2=np.array([[5,6,7],[8,9,10]])
X3=np.dot(X1,X2)
print(X3)
'''

#1.4 数组变形
#1.4.1 更改数组形状
'''
arr=np.arange(10)
print(arr)
#1.reshape不修改向量本身
print(arr.reshape(2,5)) 
print(arr.reshape(5,-1))
print(arr.reshape(-1,5))
print(arr)
#2.resize修改向量本身
print(arr.resize(2,5)) 
print(arr)
'''
'''
#3.T转置（不修改向量本身）
arr=np.arange(12).reshape(3,4)
print(arr)
print(arr.T)
print(arr)

#4.ravel向量展平
arr=np.arange(6).reshape(2,-1)
print(arr)
print(arr.ravel('F'))
print(arr.ravel()) #默认行优先，别忘了内层有个（）
print(arr)

#5.flatten将矩阵转换为向量
a=np.floor(10*np.random.random((3,4)))
print(a)
print(a.flatten())
print(a)

#6.squeeze降维（将矩阵中含1的维度去掉）
arr=np.arange(3).reshape(3,1)
print(arr)
print(arr.squeeze())
arr1=np.arange(6).reshape(3,1,2,1)
print(arr1)
print(arr1.squeeze())

arr2=np.arange(24).reshape(2,3,4)
print(arr2.shape)#从外往里剥
print(arr2.transpose(1,2,0).shape)
print(arr2)
print(arr2.transpose(1,2,0))


#1.4.2合并数组
#1.append合并一维
a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.append(a,b)
print(c)

a=np.arange(4).reshape(2,2)
b=np.arange(4).reshape(2,2)
c=np.append(a,b,axis=0)
d=np.append(a,b,axis=1)
e=np.append(a,b)
print(a)
print(b)
print(c)
print(d)
print(e)

#2.concatenate沿指定轴连接数组或矩阵
a=np.array([[1,2],[3,4]])
b=np.array([[5,6]])
c=np.concatenate((a,b),axis=0)
d=np.concatenate((a,b.T),axis=1)
print(a)
print(b)
print(b.shape)
print(c)
print(d)

#3.stack沿指定轴堆叠数组或矩阵
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
print(np.stack((a,b),axis=0))


#1.5批处理
data_train=np.random.randn(1000,2,3)
print(data_train.shape)
np.random.shuffle(data_train)
batch_size=100
for i in range(0,len(data_train),batch_size):
    x_batch_sum=np.sum(data_train[i:i+batch_size])
    print("第{}批次，该批次的数据之和：{}".format(i,x_batch_sum))


#1.6通用函数
#1.math与numpy函数的性能比较
import time
import math

x=[i*0.001 for i in np.arange(1000000)]
start=time.clock()#统计CPU时间，常用于统计某一程序或函数的执行速度
for i,t in enumerate(x):#enumerate遍历一个集合对象，遍历的同时得到当前元素的索引
    x[i]=math.sin(t)
print("math.sin:",time.clock()-start)#现在的时间-开始的时间

x=[i*0.001 for i in np.arange(1000000)]
x=np.array(x)
start=time.clock()
np.sin(x)
print("numpy.sin:",time.clock()-start)

#1.2循环与向量运算比较
import time

x1=np.random.rand(1000000)
x2=np.random.rand(1000000)
##使用循环计算向量点积
tic=time.process_time()
dot=0
for i in range(len(x1)):
    dot+=x1[i]*x2[i]
toc=time.process_time()
print("dot="+str(dot)+"\n for loop----Computation time="+str(1000*(toc-tic))+"ms")
##使用numpy函数求点积
tic=time.process_time()
dot=0
dot=np.dot(x1,x2)
toc=time.process_time()
print("dot="+str(dot)+"\n for verctor version----Computation time="+str(1000*(toc-tic))+"ms")
'''
'''
#1.7广播机制
A=np.arange(0,40,10).reshape(4,1)
B=np.arange(0,3)
C=A+B
print(A)
print(B)
print(C)
print(A.shape)
print(B.shape)
print(C.shape)
'''
