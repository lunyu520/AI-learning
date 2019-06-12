import numpy as np
import matplotlib.pyplot as plt
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 读取数据
x=[1.5,0.8,2.6,1.0,0.6,2.8,1.2,0.9,0.4,1.3,1.2,2.0,1.6,1.8,2.2]
y=[3.1,1.9,4.2,2.3,1.6,4.9,2.8,2.1,1.4,2.4,2.4,3.8,3.0,3.4,4.0]

# train_data = np.loadtxt('train_fish.txt', delimiter=',')
# test_data = np.loadtxt('test_fish.txt', delimiter=',')
# 提取训练集数据
train_X = train_data[:, :-1]
train_y = train_data[:, -1]
# 提取测试集数据
test_X = test_data[:, :-1]
test_y = test_data[:, -1]
# 训练集初始化
train_X = np.c_[np.ones(len(train_X)), train_X]
train_y = np.c_[train_y]
# 测试集初始化
test_X = np.c_[np.ones(len(test_X)), test_X]
test_y = np.c_[test_y]
# 定义代价函数
def costFunction(X, y ,theta=[[0],[0]]):
    m = X.shape[0]
    J = 1.0/(2*m)*np.sum(np.square(X.dot(theta)-y))
    return J
# 定义梯度下降函数
def granDesc(X, y, theta=[[0],[0]], alpha=0.001, iter_num=15000):
    m = X.shape[0]
    J_history = np.zeros(iter_num)
    for i in range(iter_num):
        J_history[i] = costFunction(X, y, theta)
        deltaTheta = (1.0/m)*(X.T.dot(X.dot(theta)-y))
        theta -= alpha*deltaTheta
    return J_history, theta
# 定义精度函数
def score(X, y, theta):
    h = np.dot(X, theta)
    y_mean = y.mean()
    u = np.sum((h-y)**2)
    v = np.sum((y-y_mean)**2)
    score = 1-u/v
    return score
# 给定训练集计算的模型
J_history, theta = granDesc(train_X, train_y, alpha=0.001, iter_num=5000)
print(theta)
# 对训练集结果预测
train_h = np.dot(train_X, theta)
test_h = np.dot(test_X, theta)
print(test_h)
# 求训练集精度
train_score = score(train_X, train_y, theta)
# 求测试集精度
test_score = score(test_X, test_y, theta)
print('训练集精度%.2f'%(train_score*100)+'%')
print('测试集精度='+str(round(test_score, 2)*100)+'%')
# 画出代价函数
plt.plot(J_history)
plt.show()

plt.title('测试集精度='+str(round(test_score, 2)*100)+'%')
plt.scatter(train_y, train_y, c='g', label='实际值')
plt.scatter(train_y, train_h, c='r', marker='x', label='预测值')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.show()

plt.title('训练集精度%.2f'%(train_score*100)+'%')
plt.scatter(test_y, test_y, c='g', label='实际值')
plt.scatter(test_y, test_h, c='r', marker='x', label='预测值')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.show()
