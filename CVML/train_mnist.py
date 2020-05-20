import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 每个批次的大小
batch_size = 100
# 计算一共需要多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个占位符（placeholder）
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 再定义一个placeholder，后面设置Dropout参数
keep_prob = tf.placeholder(tf.float32)
# 定义学习率变量
lr = tf.Variable(0.001, dtype=tf.float32)

# 创建一个简单的神经网络
# 第一个隐藏层
# 权重
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))  # 截断的正态分布中输出随机值，标准差：stddev
# 偏置值
b1 = tf.Variable(tf.zeros([500]) + 0.1)
# 使用激活函数，获得信号输出，即此层神经元的输出
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 调用tensorflow封装好的dropout函数，keep_prob参数是设置有多少的神经元是工作的，在训练时，通过feed操作将确切值传入
L1_drop = tf.nn.dropout(L1, keep_prob)

# 第二个隐藏层：2000个神经元
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 最后一层输出层：10个神经元
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)
# 交叉熵代价函数（cross-entropy）的使用
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用Adam优化器方法
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
# equal中的两个值，若是一样，则返回True，否则返回False。argmax函数：返回最大值所在的索引值，即位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
# 将上一步的布尔类型转化为32位浮点型，即True转换为1.0，False转换为0.0，然后计算这些值的平均值作为准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 创建一个saver
saver = tf.train.Saver()
print(saver)
# 定义会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 迭代21个周期
    for epoch in range(11):
        # 每迭代一个周期,重新给学习率赋值，目的：在后期收敛时，防止学习率过大，因此降低学习率，使得loss达到最小
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        # n_batch:之前定义的批次
        for batch in range(n_batch):
            # 获得100张图片，图片的数据保存在batch_xs中，图片的标签保存在batch_ys中
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 使用Feed操作，此步执行训练操作的op，将数据喂给他,keep_prob设置为1.0就相当于Dropout没有起作用
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        learning_rate = sess.run(lr)
        # 训练一个周期后就可以看下准确率，使用Feed方法，此步执行计算准确度的op操作，将其对应的参数喂给它
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ", Learning Rate= " + str(learning_rate))
    saver.save(sess, "./model/ckpt")
    tf.train.write_graph(sess.graph.as_graph_def(), '.', './checkpoint/mnist.pbtxt', as_text=True)
