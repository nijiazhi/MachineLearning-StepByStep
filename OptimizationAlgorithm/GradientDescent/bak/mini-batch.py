# 引用模块
import pandas as pd
import numpy as np

# 导入数据
train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')
submit = pd.read_csv('../../data/sample_submit.csv')

# 初始设置
beta = [1, 1]
alpha = 0.2
tol_L = 0.1
batch_size = 16

# 对x进行归一化
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

# 定义计算mini-batch随机梯度的函数
def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)

# 定义更新beta的函数
def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# 定义计算RMSE的函数
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

rmse_result_list = []


# 进行第一次计算
np.random.seed(10)
grad = compute_grad_batch(beta, batch_size, x, y)
loss = rmse(beta, x, y)
rmse_result_list.append(loss)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad_batch(beta, batch_size, x, y)

    tmp_loss = rmse(beta, x, y)
    rmse_result_list.append(tmp_loss)

    if i % 100 == 0:
        loss = loss_new
        loss_new = rmse(beta, x, y)
        # rmse_result_list.append(loss_new)
        print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
    i += 1
print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))
print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))
res = rmse(beta, x, y)
print('Our RMSE: %s'%res)

# output rmse result
rmse_result_path = '../rmse_result/mini-batch-stochastic-rmse-result.csv'
rmse_result_df = pd.DataFrame(rmse_result_list)
rmse_result_df.to_csv(rmse_result_path)

print('\n', '*' * 40, '\n')
print('All done', len(rmse_result_list))