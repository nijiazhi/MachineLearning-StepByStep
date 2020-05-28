import numpy as np

'''
andy：
    - 明确hmm的三个问题，及其对应解法
    - 隐马的观测可以是多维度的一个状态（比如：一个经纬度点是一个观测；一个特征向量是一个状态）

参考博客：
    https://zhuanlan.zhihu.com/p/85454896
    https://www.cnblogs.com/pinard/p/6991852.html
'''


class HMM(object):
    def __init__(self, N, M, pi=None, A=None, B=None):
        self.N = N  # 状态数量
        self.M = M  # 观察数量
        self.pi = pi  # 初始状态概率向量
        self.A = A  # 状态转移矩阵
        self.B = B  # 观测概率矩阵

    def get_data_with_distribute(self, distribute):
        """
        根据给定的概率分布随机返回数据（数据的索引）
        :param distribute: 各个概率之和为1
        :return:
        """
        r = np.random.rand()
        for i, p in enumerate(distribute):
            if r < p:
                return i
            r -= p  # 相当于一段一段的取值

        # 用下面这句也可以,按照一定概率取那个索引
        # return np.random.choice(np.arange(len(distribute)), p=distribute)

    def generate(self, T: int):
        '''
        根据给定的参数生成观测序列
        T: 指定要生成数据的数量（时序的概念）
        '''
        z = self.get_data_with_distribute(self.pi)  # 根据初始概率分布生成第一个【状态】
        x = self.get_data_with_distribute(self.B[z])  # 生成第一个【观测】数据
        result = [x]
        for _ in range(T - 1):  # 依次生成余下的状态和观测数据
            z = self.get_data_with_distribute(self.A[z])  # 当前状态到下一状态
            x = self.get_data_with_distribute(self.B[z])  # 当前状态生成的观测值
            result.append(x)
        return result

    def evaluate(self, X):
        '''
        根据给定的参数计算条件概率
        X: 观测数据
        B：观测概率矩阵
        A：状态转移矩阵
        '''
        alpha = self.pi * self.B[:, X[0]]  # 初始alpha的计算，一个向量：每个状态对应的alpha值
        for x in X[1:]:
            # alpha_next = np.empty(self.N)
            # for j in range(self.N):
            #     alpha_next[j] = np.sum(self.A[:,j] * alpha * self.B[j,x])  # 第j个状态
            # alpha = alpha_next
            alpha = np.sum(self.A * alpha.reshape(-1, 1) * self.B[:, x].reshape(1, -1), axis=0)
        return alpha.sum()

    def evaluate_backward(self, X):
        '''
        根据给定的参数计算条件概率，反向计算条件概率
        :param X:
        :return:
        '''
        beta = np.ones(self.N)
        for x in X[:0:-1]:
            beta_next = np.empty(self.N)
            for i in range(self.N):
                beta_next[i] = np.sum(self.A[i, :] * self.B[:, x] * beta)
            beta = beta_next
        return np.sum(beta * self.pi * self.B[:, X[0]])


if __name__ == "__main__":
    pi = np.array([.25, .25, .25, .25])
    A = np.array([
        [0, 1, 0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])
    hmm = HMM(4, 2, pi, A, B)

    # 生成10个数据
    print(hmm.generate(10))

    # 0.026862016
    print(hmm.evaluate([0, 0, 1, 1, 0]))

    # 0.026862016
    print(hmm.evaluate_backward([0, 0, 1, 1, 0]))

