import tensorflow as tf
import numpy as np
import dataController
from cnnController import textCNN
from tensorflow.contrib import learn

'''参数设置'''
# 数据读取
tf.flags.DEFINE_string("posData", "./data/rt-polarity.pos", "")  # 文件路径
tf.flags.DEFINE_string("negData", "./data/rt-polarity.neg", "")  # 文件路径
tf.flags.DEFINE_float("testSet", .1,"") # 取0.1做验证集

# 神经网络模型超参数
tf.flags.DEFINE_string("filterSize", "3,4,5", "") # 每次卷积3\4\5个单词 - x个单词一卷积
tf.flags.DEFINE_integer("filterNum", 128, "")   # 每次卷积得到128个特征图,增大之后训练会变慢
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "")  # 忽略一半减少过拟合，只在训练时有用

# 训练参数
batch_size = 128
epoch_num = 100     # 默认做100轮
access = 100        # 每迭代x次评估一下效果
save_every = 200    # 每迭代x次保存模型
saveTime = 5

FLAGS = tf.flags.FLAGS  # flags.FLAGS
FLAGS.flag_values_dict()    # 解析成字典存储到FLAGS.__flags中

'''数据预处理 - 数据和label'''
text, y = dataController.loadDataAndLabel(FLAGS.posData, FLAGS.negData) # 读入洗过的数据和label

# 创建词汇表，完成从词汇表到矩阵的映射 - 找到最长的文本，填充，映射到矩阵
vacabulary = learn.preprocessing.VocabularyProcessor(max([len(x.split(" ")) for x in text]))  # 预处理填充长度
x = np.array(list(vacabulary.fit_transform(text)))   # 转换格式 - 至此，大小一致

# 洗牌，让数据均衡
indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[indices] # x shuffle后
y_shuffled = y[indices]

# 交叉验证 负数为了从后往前
index = int(FLAGS.testSet * float(len(y))) * (-1)
x_train, x_test = x_shuffled[:index], x_shuffled[index:]
y_train, y_test = y_shuffled[:index], y_shuffled[index:]

'''搭建卷积神经图网络'''
with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)   # 创建配置session 自动分配设备；是否要打印日志
    session = tf.Session(config = config)

    with session.as_default():  # 卷积神经网络整体结构
        cnn = textCNN(
            seq_length = x_train.shape[1],   # 长度
            num_classes = y_train.shape[1],   #  如果y只有两列值即二分类
            vocabularySize = len(vacabulary.vocabulary_),
            filterSize = list(map(int, FLAGS.filterSize.split(","))), # 指定filter，分割一下
            filterNum = FLAGS.filterNum,  # 一共有几个filter
        )
        '''
            定义训练程序
        '''
        global_step = tf.Variable(0, name="global_step", trainable=False)   # global_step 一共要迭代多少次
        optimizer = tf.train.AdamOptimizer(0.001)
        # 把梯度和变量显示出来
        gradientAndVariable = optimizer.compute_gradients(cnn.loss)  # 传进来loss值
        trainOperation = optimizer.apply_gradients(gradientAndVariable, global_step = global_step)   # 训练操作
        # 保存模型 最大保存多少个
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = saveTime)
        session.run(tf.global_variables_initializer())  # 全局初始化

        # 训练集一次迭代
        def train_step(x_batch, y_batch):
            feed_dict = {   # 字典结构
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = session.run([trainOperation, global_step, cnn.loss, cnn.accuracy], feed_dict)
            print("iter{}, loss : {:g}, acc : {:g}".format(step, loss, accuracy))    # 当前的loss

        # 验证集一次迭代
        def test_step(x_batch, y_batch):
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.keep_prob: 1.0    # 验证集的时候全部保留，本质上这个参数在验证集没用
            }
            step, loss, accuracy = session.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
            print("iter{}, loss : {:g}, acc : {:g}".format(step, loss, accuracy))

        # 训练是一个batch一个batch来做的
        batches = dataController.batch_iter(list(zip(x_train, y_train)), batch_size, epoch_num)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)    # 训练
            current_step = tf.train.global_step(session, global_step)
            if current_step % access == 0:  # 每隔access个batch做一次验证集
                print("\n-------- 验证集 --------\n")  # 验证集这里始终不高啊
                test_step(x_test, y_test)
                print("\n-----------------------\n")
            if current_step % save_every == 0:  # 每隔save_every个batch保存一次模型
                path = saver.save(session, './model/', global_step = current_step) # 模型保存文件夹 第几次迭代 没有这个文件夹就自己造
                print("\n--------Saved Model File to {}--------\n".format(path))

