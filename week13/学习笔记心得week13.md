#### 淘宝定向广告演化

- 定向广告VS搜索广告
  - 有无明显意图
  - 有无历史数据
  
1. LR模型
2. MLR（Mixed Logistic Model）
  - 分段线性+级联：对特征进行分段，采用不同的LR模型
3. DNN
  - 特征：user profile, user behaviors, candidate Ad, context features(设备、时间)
  - 结构：embedding+MLP
    1. embedding
    2. (sum pooling)+concat
    3. 输入MLP
  - 不足
    - 用户兴趣多样
4. DIN(Deep Interest Network)
  - 兴趣多样性
  - 局部激活：只有部分历史数据对当前行为有帮助
  - 加入Attention机制
    - 在polling的时候，与candidate Ad相关的商品权重更大，不相关的权重小 
    - Activation Unit：外积+PReLu/Dice+Linear
  - 改进AUC
    - 对每个用户单独计算自身AUC，根据行为数量进行加权处理
  - Relalmpr
    - $\frac{AUC(measured model)-0.5}{AUC(based model)-0.5}-1$
  - Dice激活函数
  - MBA-Reg：mini-batch aware regularization
5. DIEN(Deep Interest Evolution Network)
  - 兴趣存在时间演化趋势
  - Interest Extractor Layer: 
    - 使用GRU结构抽取每个时间片内的用户兴趣
    - AUGRU(Attention Update GRU):用Attention权重更新门控
  - auxiliary loss
    - 用下一时刻的行为监督当前时刻兴趣的学习，促使GRU在提炼兴趣表达上更高效

6. DSIN(Deep Session Interest Network)
  - 用户序列看作多个session
  - session内相近，session之间差别较大
  - 和Airbnb中的处理一样
  1. session division layer
  2. session interest extractor layer
  3. session interest interacting layer
  4. session interest activating layer
  - 引入self-attention网络（transformer）
  - Bi-LSTM

- 线上A/B Test效果：
  - CTR : 点击率
  - CVR : 展示广告购买情况，商家关注的指标
  - GPM : 平均1000次展示，平均成交金额的比例
  
  
##### project
天猫复购用户预测Challenge





