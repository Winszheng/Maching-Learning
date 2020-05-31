# 数据操纵文件
import numpy as np
import re

# 数据处理函数 - 不能够直接读取最原始的文件
def loadDataAndLabel(posData, negData):

    # 以rb的形式把数据读进来，用utf-8解码。至此，整个一大份文件读进来了。
    positive = open(posData, "rb").read().decode('utf-8')
    negative = open(negData, "rb").read().decode('utf-8')

    # 回车分割
    pos_set = positive.split('\n')[:-1]
    neg_set = negative.split('\n')[:-1]

    # 去除空格
    pos_set = [_.strip() for _ in pos_set]
    neg_set = [_.strip() for _ in neg_set]

    # 去掉无用信息，比如标点符号
    text = pos_set + neg_set
    text = [cleanText(sent) for sent in text]

    # pos:0,1 neg:1,0
    pos_label = [[0, 1] for _ in pos_set]
    neg_label = [[1, 0] for _ in neg_set]

    # pos label和neg label组合在一起
    y = np.concatenate([pos_label, neg_label], 0)
    return [text, y]

# 为训练提供经过shuffle的数据
def batch_iter(data, batch_size, epoch_num, shuffle = True):
    data = np.array(data)
    data_len = len(data)
    num_batch = int((len(data)-1)/batch_size) + 1   # 在一轮（epoch）中会进行多少次迭代（batch）
    for _ in range(epoch_num):  # 外层循环epoch，内层循环batch
        if shuffle:
            indices = np.random.permutation(np.arange(data_len))
            shuffled_data = data[indices]
        else:
            shuffled_data = data
        for i in range(num_batch):
            start_id = i * batch_size
            end_index = min((i + 1) * batch_size, data_len)
            yield shuffled_data[start_id:end_index]

# 清洗数据 - 正则表达式
def cleanText(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

