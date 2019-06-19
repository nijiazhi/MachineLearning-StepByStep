import pandas as pd
import numpy as np
import time


def compute_grad_batch(beta, batch_size, x, y):
    """
    回归问题，最小二乘法求解，计算梯度
    y = beta_0 + beta_1*x

    :param beta:
    :param x:
    :param y:
    :return:
    """
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
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
    batch_size = 16  # mini-batch的size
    max_iters = 500   # 数据集最大遍历次数（类似深度学习中epoch数量）

    # 读取数据
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    submit = pd.read_csv('./data/sample_submit.csv')
    print(train.head(2))

    # 对x进行归一化
    max_x = max(train['id'])
    x = train['id'] / max_x
    y = train['questions']

    rmse_result_list = []

    # 进行第一次计算
    np.random.seed(10)
    loss = rmse(beta, x, y)
    rmse_result_list.append(loss)
    grad = compute_grad_batch(beta, batch_size, x, y)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(beta, x, y)
    print('first gradient:', grad, loss, loss_new)
    print()

    # 开始迭代
    i = 1
    sample_num = x.shape[0]
    while np.abs(loss_new - loss) > tol_L and i < max_iters:
        mini_batch_num = sample_num // batch_size
        for j in range(mini_batch_num):
            # rmse_result_list.append(loss_new)
            grad = compute_grad_batch(beta, batch_size, x, y)
            beta = update_beta(beta, alpha, grad)
            loss = loss_new
            loss_new = rmse(beta, x, y)
        print('Round: %s, Diff RMSE: %s ' % (i, abs(loss_new - loss)), grad)
        rmse_result_list.append(loss_new)
        i += 1

    print('\n', '*'*40, '\n')
    print('Coef: %s \nIntercept %s' % (beta[1], beta[0]))
    print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))
    res = rmse(beta, x, y)
    print('Our RMSE: %s' % res)

    # output rmse result
    rmse_result_path = './rmse_result/mini-batch-stochastic-rmse-result.csv'
    rmse_result_df = pd.DataFrame(rmse_result_list)
    rmse_result_df.to_csv(rmse_result_path)

    print('\n', '*'*40, '\n')
    print('All done', len(rmse_result_list))


if __name__ == '__main__':
    t1 = time.time()

    main()

    t2 = time.time()
    print('\n############ %s done: | %fmin ############' % ('stochastic', (t2 - t1) / 60))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2)))

