# coding:utf8

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          show_stdv=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='barh', title='Feature Importances')
    plt.xlabel('Feature Importance Score')


if __name__ == '__main__':

    train_set = pd.read_csv('./Dataset/train_modified.csv')
    print(train_set.shape)
    target = 'Disbursed'
    IDcol = 'ID'

    # Choose all predictors except target & IDcols
    predictors = [x for x in train_set.columns if x not in [target, IDcol]]

    # 第一步：确定学习速率和tree_based 参数调优的估计器数目
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',  # 二分类任务
        nthread=3,
        scale_pos_weight=1,
        seed=27
    )
    modelfit(xgb1, train_set, predictors)

    # 第二步： max_depth 和 min_weight 参数调优
    param_test1 = {
     'max_depth': range(3, 10, 2),
     'min_child_weight': range(1, 6, 2)
    }

    grid_search_1 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=3, scale_pos_weight=1, seed=27),
        param_grid=param_test1,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    grid_search_1.fit(train_set[predictors], train_set[target])
    print(grid_search_1.cv_results_, grid_search_1.best_params_, grid_search_1.best_score_, '\n\n')

    '''
    至此，我们对于数值进行了较大跨度的12中不同的排列组合，
    可以看出理想的max_depth值为5，理想的min_child_weight值为5。在这个值附近我们可以再进一步调整，
    来找出理想值。我们把上下范围各拓展1，因为之前我们进行组合的时候，参数调整的步长是2
    '''
    param_test2 = {
     'max_depth': [4, 5, 6],
     'min_child_weight': [4, 5, 6]
    }
    gsearch2 = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                  objective= 'binary:logistic', nthread=3, scale_pos_weight=1,seed=27),
        param_grid=param_test2,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    gsearch2.fit(train_set[predictors], train_set[target])
    print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_, '\n\n')

    '''
    虽然min_child_weight的理想取值是6，但是我们还没尝试过大于6的取值。像下面这样，就可以尝试其它值。
    '''
    param_test2b = {
     'min_child_weight': [6, 8, 10, 12]
     }

    gsearch2b = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.1, n_estimators=140, max_depth=4,
            min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', nthread=3, scale_pos_weight=1,seed=27),
        param_grid=param_test2b,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)

    gsearch2b.fit(train_set[predictors], train_set[target])
    print(gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_, '\n\n')


    # 第三步：gamma参数调优
    '''
    在已经调整好其它参数的基础上，我们可以进行gamma参数的调优了。
    Gamma参数取值范围可以很大，我这里把取值范围设置为5了。你其实也可以取更精确的gamma值。
    '''
    param_test3 = {
     'gamma': [i/10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(
        estimator=XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6,
                                gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                nthread=3, scale_pos_weight=1,seed=27),
        param_grid = param_test3,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)

    gsearch3.fit(train_set[predictors], train_set[target])
    print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_, '\n\n')
    modelfit(gsearch3.best_estimator_, train_set, predictors)

    '''
    从这里可以看出来，我们在第一步调参时设置的初始gamma值就是比较合适的。
    也就是说，理想的gamma值为0。在这个过程开始之前，最好重新调整boosting回合，因为参数都有变化。
    '''
    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb2, train_set, predictors)


    # 第四步：调整subsample 和 colsample_bytree 参数
    param_test4 = {
     'subsample': [i/10.0 for i in range(6, 10)],
     'colsample_bytree': [i/10.0 for i in range(6, 10)]
    }

    gsearch4 = GridSearchCV(
        estimator=XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=3, min_child_weight=4, gamma=0.1,
                                subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=3,
                                scale_pos_weight=1,seed=27),
        param_grid = param_test4,
        scoring='roc_auc',
        n_jobs=2,
        iid=False, cv=5)

    gsearch4.fit(train_set[predictors], train_set[target])
    print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_, '\n\n')
    '''
    从这里可以看出来，subsample 和 colsample_bytree 参数的理想取值都是0.8。
    现在，我们以0.05为步长，在这个值附近尝试取值
    '''

    param_test5 = {
     'subsample': [i/100.0 for i in range(75, 90, 5)],
     'colsample_bytree': [i/100.0 for i in range(75, 90, 5)]
    }

    gsearch5 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0,
            subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=3, scale_pos_weight=1,seed=27),
        param_grid = param_test5,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)

    gsearch5.fit(train_set[predictors], train_set[target])

    # 第五步：正则化参数调优
    '''
    下一步是应用正则化来降低过拟合。
    由于gamma函数提供了一种更加有效地降低过拟合的方法，大部分人很少会用到这个参数。
    但是我们在这里也可以尝试用一下这个参数。我会在这里调整’reg_alpha’参数，然后’reg_lambda’参数留给你来完成。
    '''
    param_test6 = {
     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    gsearch6 = GridSearchCV(
        estimator=XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0.1,
                                subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=3,
                                scale_pos_weight=1,seed=27),
        param_grid=param_test6,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)

    gsearch6.fit(train_set[predictors], train_set[target])
    print(gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_)
    '''
    我们可以看到，相比之前的结果，CV的得分甚至还降低了。
    但是我们之前使用的取值是十分粗糙的，我们在这里选取一个比较靠近理想值(0.01)的取值，来看看是否有更好的表现。
    '''
    param_test7 = {
     'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
    }
    gsearch7 = GridSearchCV(
        estimator=XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6,
                                gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                nthread=3, scale_pos_weight=1,seed=27),
        param_grid = param_test7,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)

    gsearch7.fit(train_set[predictors], train_set[target])
    print(gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_)

    '''
    可以看到，CV的得分提高了。现在，我们在模型中来使用正则化参数，来看看这个参数的影响。
    '''
    xgb3 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=4,
     min_child_weight=6,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     reg_alpha=0.005,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
    modelfit(xgb3, train_set, predictors)

    xgb4 = XGBClassifier(
        learning_rate=0.01,  # 改变学习率
        n_estimators=5000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb4, train_set, predictors)

    print('\n\nALL DONE\n\n')
    plt.show()
