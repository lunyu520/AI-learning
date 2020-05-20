import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


class simpleInfer(object):
    def __init__(self, model_path):
        self.load_model(model_path)
        self.sess = None

    def load_model(self, model_path):
        # 定义两个占位符（placeholder）
        self.x = tf.placeholder(tf.float32, [None, 784])
        # 再定义一个placeholder，后面设置Dropout参数
        keep_prob = tf.placeholder(tf.float32)
        # 权重
        W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))  # 截断的正态分布中输出随机值，标准差：stddev
        # 偏置值
        b1 = tf.Variable(tf.zeros([500]) + 0.1)
        # 使用激活函数，获得信号输出，即此层神经元的输出
        L1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
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
        self.prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)
        # 创建一个saver
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)
        saver.restore(self.sess,model_path)

    def infer(self, inp):
        """Infer method
        inputs:
        - inp: list, a list of np array, with size (28, 28)
        returns:
        - list, a list of int, which is the result of the model inference
        """
        assert self.sess is not None, "Model is not loaded!"
        ys = []
        for x_test in inp:
            y1 = tf.argmax(self.sess.run(self.prediction, feed_dict={self.x: x_test}), 1).eval()
            ys.append(y1)
        return ys


if __name__ == '__main__':
    testinf = simpleInfer("./model/ckpt")
    for i in range(100):
        # 每次测试一张图片 [0,0,0,0,0,1,0,0,0,0]
        x_test, y_test = mnist.test.next_batch(1)
        print(x_test)
        print("第%d张图片，手写数字图片目标是:%d, 预测结果是:%d" % (
            i,
            tf.argmax(y_test, 1).eval(),
            testinf.infer([x_test])[0]
        ))
