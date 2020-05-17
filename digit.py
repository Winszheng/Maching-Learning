# 全连接神经网络实现手写数字识别
import tensorflow as tf
import numpy as np

# down the mnist folder
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST',one_hot=True)   #hot:1 cold:0; predict:0-9 10位向量 one_hot: 有且仅有一个不是0 eg.1000000000 - 0
tf.logging.set_verbosity(old_v)

# input layer; placeholder:占位符，先搭结构，值以后再说
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name = 'x') #784:图片size，None:数量未知
# output layer
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name = 'y')
batch_size = 128

# 添加每一层的方法
def add_layer(input_data, input_num, output_num, activation_function = None):
    # output = input_data*weight + bias 线性变换
    w = tf.Variable(initial_value=tf.random_normal(shape=[input_num, output_num]), trainable=True)  #bg过程中优化
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, output_num]))
    output = tf.add(tf.matmul(input_data,w), b)
    # activation ? output+activation_function(output) : output 非线性变换
    if activation_function:
        output = activation_function(output)
    return output

# 搭建整个神经网络 - sigmoid
def build_nn(data):
    hidden_layer1 = add_layer(data, 784, 100, tf.nn.sigmoid)    #输入值，输入大小，输出大小，激活函数
    hidden_layer2 = add_layer(hidden_layer1, 100, 50, tf.nn.sigmoid)
    output_layer = add_layer(hidden_layer2, 50, 10)
    return  output_layer


# 训练网络 - 思路： output和y做比较，用loss做判断
def train_nn(data):
    # output of NN
    output = build_nn(data) 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    # 优化器
    lr = tf.Variable(0.01,dtype=tf.float32)
    optimizer =  tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        # 初始化前面的变量
        sess.run(tf.global_variables_initializer())
        for i in range(40):
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                x_data, y_data = mnist.train.next_batch(batch_size)
                cost, _ = sess.run([loss,optimizer], feed_dict={x:x_data, y:y_data})
                epoch_cost += cost
            print('epoch_cost ', i, ' : ', epoch_cost)
        #准确度
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(output, 1)),tf.float32))
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('       accuracy : ', acc)

#调用
train_nn(x)


