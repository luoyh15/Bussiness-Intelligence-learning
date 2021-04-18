import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pickle

if __name__ == '__main__':
    # load data
    path = 'data/lgb/'
    train_X = pd.read_pickle(path+'train_X.pkl')
    test_X = pd.read_pickle(path+'test_X.pkl')
    train_y = pd.read_pickle(path+'train_y.pkl')

    param = {'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        "lambda_l2": 2,  # 防止过拟合
        'min_data_in_leaf': 20,  # 防止过拟合
        'objective': 'regression_l1',
        'learning_rate': 0.02,

        "feature_fraction": 0.8,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": 2011,
        "metric": 'mae',
        }
    # 类别列
    categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'city', 'regionCode']
    
    fetures = train_X.columns
    # 加载特征重要性（倒序排列）
    with open('feature_importance.pkl', 'rb') as fp:
        featurs = pickle.load(fp)
    # 只取前70个重要特征
    featurs = featurs[-50:]
    # 去掉不重要的列
    categorical_cols = [col for col in categorical_cols if col in featurs]
    
    train_X, test_X = train_X[featurs], test_X[featurs]
    
    train_data = lgb.Dataset(train_X, train_y, categorical_feature=categorical_cols)

    model = lgb.cv(param, train_data, num_boost_round=100000, early_stopping_rounds=300, verbose_eval=300, return_cvbooster=True,eval_train_metric=True, nfold=5)


    # 先用一个模型得到特征重要性
    # model = lgb.train(param, train_data, num_boost_round=100000, valid_sets=train_data,early_stopping_rounds=300, verbose_eval=300)
    # importance = lgb.plot_importance(model, figsize=(10, 20))
    # ytick_labels = importance.get_yticklabels()
    # features = list(map(lambda x: x.get_text(), ytick_labels))
    # with open('feature_importance.pkl', 'wb') as fp:
    #     pickle.dump(features, fp)
    # plt.savefig("importance_lgb.png")

    # 预测
    predicts = model['cvbooster'].predict(test_X)

    submission = pd.read_csv('data/used_car_sample_submit.csv')
    submission['price'] = np.array(predicts).mean(axis=0)
    submission.to_csv('lgb_submission.csv')