#### 时间序列模型
- 目标变量与时间的关系
- 用途
  - 用于预测未来（十分有用，数据驱动）
  - 异常检测
- 机器学习模型：
  - AR，MA，ARMA，ARIMA (实为统计模型)
- 神经网络模型
  - RNN，LSTM模型，GRU，Transformer（目前主流）
  
- 时间序列
  - 平稳序列（无趋势、无周期）
  - 非平稳序列
  - 特征
    - 趋势(长期)
    - 季节性(短周期)
    - 周期性(>1年)
    - 随机性

- 基于统计的模型
    - AR模型
      - Auto Regressive：历史值的线性组合
    - MA模型
      - Moving Average：历史噪声的线性组合
    - ARMA：AR+MA
    - ARIMA: AR Integrated MA
      - 不平稳数据进行差分处理
    - 只有时间一个维度，信息过少，不够准确。
    
- 基于神经网络
  - RNN循环神经网络
  - LSTM解决RNN梯度消失的问题
    - 记忆门、遗忘门、输出门
  
- 工具
  - statsmodels
    - 包含:
      - 回归模型
      - 方差分析（ANOVA）
      - 时间序列（AR，ARMA，ARIMA）
    - tsa：time series analysis
      - seasonal_decompose:做分解
        - trend（趋势），seasonal（季节性）和residual (残留）
      - ARMA：
        - endog:endogenous variable 内生变量：非政策性
        - order: p,q
        - exog：exogenous variable 外生变量：政策性
    - 评价指标：AIC 赤池消息准则 (简单理解为loss function)
  - keras(LSTM)
    - 多个特征
    - 训练时间长
    - 不能采用train_test_split() 因为时间序列不连续（XGBoost可以，因为样本是相互独立的）
  
  
#### project 相关
- 步骤
  1. 数据加载和探索
    - 选择合适的时间尺度
    - 数据归一化
    - 构造有监督学习数据
  2. 模型选择和训练
    - 最优模型：grid search
    - 容易过拟合、惯性、延迟效应
    - 噪音过大
  3. 预测和可视化


    
    
      