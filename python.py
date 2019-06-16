import numpy as np
from numpy import *
import matplotlib.pylab as plt
from pylab import *
# 显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 显示负号
matplotlib.rcParams['axes.unicode_minus'] = False
# 加载数据
train_data = np.loadtxt('box_train.txt', delimiter=',')
test_data = np.loadtxt('box_test.txt', delimiter=',')
# 提取训练集数据
train_X = train_data[:, 0:2]
train_y = train_data[:, -1]
# 提取测试集数据
test_X = test_data[:, 0:2]
test_y = test_data[:, -1]
# 训练集特征缩放
mu = np.mean(train_X, 0)
sigma = np.std(train_X, 0)
train_X -= mu
train_X /= sigma
# 测试集特征缩放
mu = np.mean(test_X, 0)
sigma = np.std(test_X, 0)
test_X -= mu
test_X /= sigma
# 训练集初始化
m = train_X.shape[0]
train_X = np.c_[np.ones(m), train_X]
train_y = np.c_[train_y]
# 测试集
n = test_X.shape[0]
test_X = np.c_[np.ones(n), test_X]
test_y = np.c_[test_y]
# 定义sigmoid函数
def sigmoid(Z):
    return 1.0/(1.0+np.exp(-Z))
# 定义代价函数
def costFuntion(X, y, theta, lamda):
    m, f = train_X.shape
    h = sigmoid(np.dot(X, theta))
    R = lamda/(2.0*m)*np.dot(theta.T, theta)
    J = -1.0/m*(np.dot(y.T, log(h))+np.dot((1-y).T, log(1-h)))+R
    return J
# 定义梯度下降函数
def granDesc(X, y, lamda, alpha=0.01, iter_num=15000):
    m, f = train_X.shape
    theta = np.zeros((f, 1))
    J_history = np.zeros(iter_num)
    for i in range(iter_num):
        J_history[i] = costFuntion(X, y, theta, lamda)
        h = sigmoid(np.dot(X, theta))
        deltatheta = 1.0/m*np.dot(X.T, (h-y))+lamda/m*theta
        theta -= alpha*deltatheta
    return J_history, theta
J_history, theta = granDesc(train_X, train_y, lamda=0)
J_history1, theta1 = granDesc(train_X, train_y, lamda=3.3)
# 定义准确率函数
def logis(X, y, theta):
    m, f = X.shape
    y_p = np.zeros(m)
    count = 0
    for i in range(m):
        h = sigmoid(np.dot(X[i, :], theta))
        if h >= 0.5:
            y_p[i] = 1
        else:
            y_p[i] = 0
        if np.where(h >= 0.5, 1, 0) == y[i]:
            count += 1
    accury = count/m
    return y_p, accury
y_p, accury = logis(train_X, train_y, theta)
print('训练集预测值', y_p)
print('训练集准确率={0}%'.format(accury*100))
y_p1, accury1 = logis(test_X, test_y, theta)
print('测试集预测值', y_p1)
print('测试集准确率={0}%'.format(accury1*100))
# 画sigmoid函数
x = np.linspace(-5, 5, 200)
plt.title('sigmoid函数')
plt.plot(x, sigmoid(x))
plt.show()
# 画代价曲线
plt.title('代价曲线')
plt.xlabel('迭代次数')
plt.ylabel('代价值')
plt.plot(J_history, 'r')
plt.plot(J_history1, 'g')
plt.show()
# 测试集分界线
for i in range(n):
    if test_y[i] == 1:
        plt.scatter(test_X[i, 1], test_X[i, 2], c='r', marker='x')
    else:
        plt.scatter(test_X[i, 1], test_X[i, 2], c='g', marker='x')
min_x = min(test_X[:, 1])
max_x = max(test_X[:, 1])
min_y = -(theta[0]+theta[1]*min_x)/theta[2]
max_y = -(theta[0]+theta[1]*max_x)/theta[2]
min_y1 = -(theta1[0]+theta1[1]*min_x)/theta1[2]
max_y1 = -(theta1[0]+theta1[1]*max_x)/theta1[2]
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot([min_x, max_x], [min_y, max_y], c='r')
plt.plot([min_x, max_x], [min_y1, max_y1], c='g')
plt.show()
