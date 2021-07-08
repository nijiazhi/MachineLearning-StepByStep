import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import samples_generator

# 记录程序运行时间
import time
start_time = time.time()

# 生成数据
X, y = samples_generator.make_classification(n_classes=10, n_samples=100000,
                                             n_features=100, n_informative=20, n_redundant=0, random_state=42)
print(type(X), X.shape, y.shape)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=1)
print(type(X_train), X_train.shape)



# xgb 矩阵赋值
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_validation = xgb.DMatrix(X_validation, label=y_validation)
xgb_test = xgb.DMatrix(X_validation)
print(type(xgb_train), xgb_train.num_row)

# xgboost模型
params = {
    'booster': 'gbtree',
    'num_class': 10,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    #  这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，
    # min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 over fitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,  # 如同学习率
    'seed': 1000,
    'nthread': 4,  # cpu 线程数
    'objective': 'multi:softmax',  # 多分类的问题
    # 'eval_metric': 'auc'
}

params_list = list(params.items())
num_rounds = 5000  # 迭代次数
watchlist = [(xgb_train, 'train_set'), (xgb_validation, 'val')]

# 训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(params_list, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
model.save_model('./model/xgb.model')  # 用于存储训练出的模型
print("best best_ntree_limit", model.best_ntree_limit)


# 预测并保存
preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

np.savetxt('xgb_submission.csv', np.c_[range(1, len(y_train)+1), preds],
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')

# 输出运行时长
cost_time = time.time() - start_time
print("xgboost success!", '\n', "cost time:", cost_time, "(s)......")


'''
    xgboost进阶：自定义 优化目标 和 评价函数
'''


def precision_recall_curve(labels, preds, pos_label=0):
    return None, None, None


def maxRecall(preds,dtrain):  # preds是结果（概率值），dtrain是个带label的DMatrix
    labels = dtrain.get_label()  # 提取label
    preds = 1-preds
    precision, recall, threshold = precision_recall_curve(labels, preds, pos_label=0)
    pr = pd.DataFrame({'precision': precision,'recall': recall})
    return 'Max Recall:', pr[pr.precision >= 0.97].recall.max()

# bst = xgb.train(param,xg_train,n_round,watchlist,feval=maxRecall,maximize=False)


# 别人的自定义损失函数, 需要自己求梯度，和二阶导数
def custom_loss(y_pre, D_label):
    label = D_label.get_label()
    penalty = 2.0
    grad = -label/y_pre+penalty*(1-label)/(1-y_pre)  # 梯度
    hess = label/(y_pre**2)+penalty*(1-label)/(1-y_pre)**2  # 2阶导
    return grad, hess

# 只要再加上obj=custom_loss就可以了。
# bst=xgb.train(param,xg_train,n_round,watchlist,feval=maxRecall,obj=custom_loss,maximize=False)
