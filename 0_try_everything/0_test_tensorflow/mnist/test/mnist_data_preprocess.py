# coding : utf-8
import pickle
import gzip


# mnist_file_path = './tensorflow/mnist/mnist_data/mnist.pkl.gz'
mnist_file_path = '../mnist_data/mnist.pkl.gz'


def load_data():
    with gzip.open(mnist_file_path) as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
    return training_data, valid_data, test_data

# training_data, valid_data, test_data 均是二元tuple
# tuple of ndarray：feature and label
training_data, valid_data, test_data = load_data()

print(len(training_data[0]))  # 50000
print(len(valid_data[0]))  # 10000
print(len(test_data[0]))  # 10000
print(len(training_data[0][0]))  # 784

from PIL import Image
pic = training_data[0][0]  # 一张图片28*28，拉成一维向量784
print(pic)
pic.resize((28, 28))
im = Image.fromarray((pic*256).astype('uint8'))
im.show()  # 显示手写数别图片（数字5）
