import pandas as pd
import numpy as np
import time

'''
参考资料：
1. http://sofasofa.io/tutorials/python_gradient_descent/2.php
2. https://blog.csdn.net/u012328159/article/details/80252012
'''


def compute_grad(beta, x, y):
    """
    回归问题，最小二乘法求解，计算梯度
    y = beta_0 + beta_1*x

    :param beta:
    :param x:
    :param y:
    :return:
    """
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)


def update_beta(beta, alpha, grad):
    """
    梯度下降，更新模型参数beta
    梯度会逐渐变小，渐变到0

    :param beta:
    :param alpha: 学习率
    :param grad:  梯度
    :return:
    """
    new_beta = np.array(beta) - alpha * grad
    return new_beta


def rmse(beta, x, y):
    """
    计算RMSE的函数

    :param beta:
    :param x:
    :param y:
    :return:
    """
    y_hat = beta[0] + beta[1] * x
    squared_err = (y_hat - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


def main():

    # 超参数
    beta = [1, 1]  # 梯度下降初始点
    alpha = 0.1    # 学习率
    tol_L = 0.1    # 迭代终止的误差忍耐度（更切合实际的作法是设置对于损失函数的变动的阈值tol_L）
    max_iters = 500  # 数据集最大遍历次数（类似深度学习中epoch数量）

    # 读取数据
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    submit = pd.read_csv('../data/sample_submit.csv')
    print(train.shape, train.head(2))

    # 对x进行归一化
    max_x = max(train['id'])
    x = train['id'] / max_x
    y = train['questions']

    rmse_result_list = []

    # 进行第一次计算
    grad = compute_grad(beta, x, y)
    loss = rmse(beta, x, y)
    rmse_result_list.append(loss)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(beta, x, y)
    print('first gradient:', grad, loss, loss_new)
    print()

    # 开始迭代
    i = 1
    while np.abs(loss_new - loss) > tol_L and i < max_iters:
        rmse_result_list.append(loss_new)
        grad = compute_grad(beta, x, y)
        beta = update_beta(beta, alpha, grad)
        loss = loss_new
        loss_new = rmse(beta, x, y)
        print('Round: %s, Diff RMSE: %s ' % (i, abs(loss_new - loss)), grad)
        i += 1

    print('\n', '*'*40, '\n')
    print('Coef: %s \nIntercept %s' % (beta[1], beta[0]))
    print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))
    res = rmse(beta, x, y)
    print('Our RMSE: %s' % res)

    ##########################
    ## 与sklearn做比较
    ##########################
    print('\n', '*'*40, '\n')
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(train[['id']], train[['questions']])
    print('Sklearn Coef: %s' % lr.coef_[0][0])
    print('Sklearn Coef: %s' % lr.intercept_[0])

    res = rmse([936.051219649, 2.19487084], train['id'], y)
    print('Sklearn RMSE: %s' % res)

    # output rmse result
    rmse_result_path = './rmse_result/full-batch-rmse-result.csv'
    rmse_result_df = pd.DataFrame(rmse_result_list)
    rmse_result_df.to_csv(rmse_result_path)
    print('\n', '*'*40, '\n')
    print('All done', len(rmse_result_list))


if __name__ == '__main__':
    t1 = time.time()

    main()

    t2 = time.time()
    print('\n############ %s done: | %fmin ############' % ('full-batch', (t2 - t1) / 60))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2)))

