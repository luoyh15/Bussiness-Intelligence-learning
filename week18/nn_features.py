import pandas as pd
import numpy as np
from datetime import datetime as dt

# 处理时间数据的函数（有异常值：月份=0）
def process_time(date_int):
    year = date_int//10000
    month = (date_int%10000)//100
    day = date_int%100
    if month <= 0 or month > 12:
        month = 1
    return dt(year,month, day)

# 处理notRepaireDamage列（有异常值：-）
def not_repaire_damage(s):
    if s=='0.0':
        return 0
    if s=='1.0':
        return 2
    return 1

# 分组建立统计特征
def group_statistic(df, group_cols):
    result = df
    agg_fun = [len, 'min', 'max', 'median', 'mean', 'std']
    #  'sum', 'skew', pd.Series.kurt, 'mad']
    for col in group_cols:
        group = df[[col,'price']].groupby(col)
        agg = group.agg(agg_fun)
        agg.columns = [col+'_'+fun[1] for fun in agg.columns]
        result = result.merge(agg.fillna(0), how='left', on=col)
    return result


if __name__ == '__main__':
    # 数据集加载
    path = 'data/'
    train_data = pd.read_csv(path+'used_car_train_20200313.csv', sep=' ')
    test_data = pd.read_csv(path+'used_car_testB_20200421.csv', sep=' ')

    # print(train_data['regionCode'])
    # 异常值处理
    train_data.drop(train_data[train_data['seller'] == 1].index, inplace=True)
    # 训练集标签
    train_y = train_data['price']

    # price是长尾分布，box-cox变换后面做统计特征
    train_data['price'] = train_data['price'].apply(np.log1p)

    # 拼接训练集和测试集一起做特征处理
    data = pd.concat([train_data, test_data])

    '''数据格式处理部分'''
    # 缺失值处理
    data['model'].fillna(0, inplace=True)
    data['bodyType'].fillna(0, inplace=True)
    data['fuelType'].fillna(0, inplace=True)
    data['gearbox'].fillna(0, inplace=True)
    # 时间列处理
    for col in ['regDate', 'creatDate']:
        data[col] = data[col].apply(process_time)
        data[col+'_month'] = data[col].dt.month
        data[col+'_year'] = data[col].dt.year
    # notRepairedDamage列处理
    data['notRepairedDamage'] = data['notRepairedDamage'].apply(not_repaire_damage)

    '''特征工程'''
    # 时间相关
    data['car_age_day'] = (data['creatDate']-data['regDate']).dt.days
    data['car_age_year'] = np.round(data['car_age_day']/365)
    # 地区相关
    data['city'] = data['regionCode'].apply(lambda x : x//100)

    # 需要统计特征的列
    group_cols = ['model', 'brand', 'regionCode']
    # 进行特征统计
    data = group_statistic(data, group_cols)

    # 不需要的列
    drop_col = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType',  'price', 'regionCode']
    # 类别列
    categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'city', 'notRepairedDamage']
    # 数值特征列
    numerical_cols = [col for col in data.columns if col not in drop_col and col not in categorical_cols]
    X_data = data[numerical_cols]
    # 数值特征做归一化
    X_data = (X_data-X_data.min())/(X_data.max()-X_data.min()+np.exp(-10))

    # 加入类别特征，这里做label encoding
    data[categorical_cols] = data[categorical_cols].astype('category').apply(lambda x: x.cat.codes)
    print(data[categorical_cols].describe(), data[categorical_cols].nunique()) 
    X_data = pd.concat([X_data, data[categorical_cols]], axis=1)


    # 训练集、测试集拆分
    train_X = X_data[data['price'].notna()]
    test_X = X_data[data['price'].isna()]

    nn_path = path + 'nn/'
    train_X.to_pickle(nn_path+'train_X.pkl')
    test_X.to_pickle(nn_path+'test_X.pkl')
    train_y.to_pickle(nn_path+'train_y.pkl')