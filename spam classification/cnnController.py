import tensorflow as tf
'''
    搭建一个用于文本数据的卷积神经网络
    词嵌入层（输入层） - 卷积层 - 池化层 - 分类
'''
class textCNN(object):
    def __init__(self, seq_length, num_classes, vocabularySize, filterSize, filterNum, l2_reg_lambda = 0.001
                 ):  # 定义初始化函数
        embedding_size = 128
        pooled_output = []

        # 词嵌入层
        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # label
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")   # 保留率
        with tf.device('/cpu:0'), tf.name_scope("embedding"):   # 强制使用cpu 数据转换成128层
            W = tf.Variable(tf.random_uniform([vocabularySize, embedding_size], -1.0, 1.0), name="W")    # 随机初始化 把一万多维转换成128维
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)   # 查找input_x当中的所有ids，获取它们的词向量
            self.embedded_chars_append = tf.expand_dims(self.embedded_chars, -1) # 增加一个维度 转换成4维 默认等于-1

        # 卷积层 + 池化层
        for i, filter_size in enumerate(filterSize):
            with tf.name_scope("convolution-maxpool-%s" % filter_size):    # 第几次了
                # 卷积层
                filterShape = [filter_size, embedding_size, 1, filterNum]    # 4维 长，宽， 输出128个特征图
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W") # 弄清楚filter_shape是为了初始化权重W 高斯初始化 0.1可以改
                b = tf.Variable(tf.constant(0.1, shape=[filterNum]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_append, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")  # strides:步长 VALID:不加padding 至此完成一次卷积
                # 加激活层
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 加池化层 ksize：池化大小
                pooled = tf.nn.max_pool(h, ksize=[1, seq_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_output.append(pooled)   # 存每一次的结果，至此，所有filter_size都进行了卷积和max-pooling

        # 把相同filter_size的所有pooled结果concat起来，再将不同concat起来，最后得到的类似二维的数组
        filterTotal = filterNum * len(filterSize)
        self.pool = tf.concat(pooled_output, 3)
        self.pool_flat = tf.reshape(self.pool, [-1, filterTotal]) # 拉平for全连接层,至此可以开始分类了

        # dropout针对隐藏层的输出层drop
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.pool_flat, self.keep_prob)

        # 输出层 - 全连接层 - 全连接层要有l2损失 - 缓解过拟合
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape = [filterTotal, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")    # 二分类
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.matmul(self.h_drop, W) + b
            self.prediction = tf.argmax(self.scores, 1, name="prediction")    # 得到预测值，谁预测高就是谁，至此，output层写完了

        # 平均交叉熵损失 - 多分类
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  + l2_reg_lambda * l2_loss

        # 平均准确率
        with tf.name_scope("accuracy"):
            correct = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")


