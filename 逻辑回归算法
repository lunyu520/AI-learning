import numpy as np
import matplotlib.pylab as plt
from numpy import *
from pylab import *
# 显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 显示负号
matplotlib.rcParams['axes.unicode_minus'] = False
# 加载数据
train_data = np.loadtxt('train.txt', delimiter=',')
test_data = np.loadtxt('test.txt', delimiter=',')

# 提取训练集数据
train_X = train_data[:, 0:3]
train_y = train_data[:, -1]
# 提取测试集数据
test_X = test_data[:, 0:3]
test_y = test_data[:, -1]

# 特征缩放训练集
mu = np.mean(train_X, 0)
sigma = np.std(train_X, 0)
train_X -= mu
train_X /= sigma
# mu = np.mean(test_X, 0)   # 测试集
# sigma = np.std(test_X, 0)
# test_X -= mu
# test_X /= sigma

# 训练集初始化
m = train_X.shape[0]
train_X = np.c_[np.ones(m), train_X]
train_y = np.c_[train_y]
n = test_X.shape[0]
test_X = np.c_[np.ones(n), test_X]
test_y = np.c_[test_y]

# 定义sigmoid函数
def sigmoid(Z):
    return 1.0/(1.0+np.exp(-Z))
# 定义代价函数
def costFunction(X, y, theta):
    m, f = train_X.shape
    h = sigmoid(np.dot(X, theta))
    J = -1.0/m*(np.dot(y.T, log(h))+np.dot((1-y).T, log(1-h)))
    return J
# 定义梯度下降函数
def granDesc(X, y, alpha=0.01, iter_num=15000):
    m, f = train_X.shape
    theta = np.zeros((f, 1))
    J_hisotory = np.zeros(iter_num)
    for i in range(iter_num):
        J_hisotory[i] = costFunction(X, y, theta)
        h = sigmoid(np.dot(X, theta))
        deltatheta = 1.0/m*np.dot(X.T, (h-y))
        theta -= alpha*deltatheta
    return J_hisotory, theta
J_hisotory, theta = granDesc(train_X, train_y)

# 定义准确率函数
def testLogis(X, y, theta):
    m, f = X.shape
    count = 0
    for i in range(m):
        h = sigmoid(np.dot(X[i, :], theta))
        if np.where(h >= 0.5, 1, 0) == y[i]:
            count += 1
    accury = count/m
    return accury
# 测试集准确率
accury = testLogis(test_X, test_y, theta)
print('测试集准确率', accury*100, '%')
# 化sigmoid函数图
x = np.linspace(-5, 5, 200)
plt.title('sigmoid函数')
plt.plot(x, sigmoid(x))
plt.show()

# 代价曲线
plt.title('代价曲线')
plt.xlabel('迭代次数')
plt.ylabel('代价值')
plt.plot(J_hisotory)
plt.show()
