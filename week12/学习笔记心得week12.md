### 个性化推荐

实时个性化：
- 搜索排序
- 相似物品

**List Embedding**
- 固定长度，可计算
- 借鉴word2vec的skip-gram方法
  - 构建NN预测上下文
  - 这个假想任务训练得到的隐藏层输出就是embedding
  - 隐藏层输出:lookup table.查找隐藏层。
- 用skip-gram构造基础目标函数
  - $L=\sum\limits_{s\in S}\sum\limits_{l_i\in s}\sum\limits_{-m\le j\le m, j\ne 0} \log P(l_{i+j}|l_i)$
- negative sampling由于这里没有负样本，需要自己构造
  - $Loss = -\sum\limits_{(l,c)\in D_p}\log\frac{1}{1+e^{-v\prime_cv^l}}-\sum\limits_{(l,c)\in D_n}\log\frac{1}{1+e^{v\prime_cv^l}}$ 
  - $v_l$ for center listing, $v_c$ for aroud the center listing
  - $D_p$ for positive sample, $D_n$ for negative sample.
- 添加booking listing
  - $Loss += -\log\frac{1}{1+e^{-v\prime_{l_b}v_l}}
  - $v_{l_b}$ for booking list$
- 适配聚集搜索(Adapting Training for Congregated Search)
  - 根据地区优化目标函数
  - 在central listing 同一区域集合中进行负采样
  - $Loss += -\sum\limits_{(l,m_n)\in D_{m_n}}\log\frac{1}{1+e^{v\prime_{m_n}v_l}}$
  - $v_{m_n}$ for negative listing in the same area of positive listing
- 冷启动
  - 用静态数据找到相似listing，加权平均作为embedding。
- 评估
  - 用K-means将embedding进行聚类，发现其可解释性。
  - 余弦相似度，发现其可解释性。 
- 不能解决的问题：
  - 基于点击
  - 只提取了short-term
  - 只能用于找相似 
- Type Embedding(归类，用整个类别的行为来进行建模)分桶
  - user type embedding
  - listing type embedding
  - 用自定义规则来分桶，分完在计算embedding
  - $Loss = -\sum\limits_{(u_t,c)\in D_{book}}\log\frac{1}{1+e^{-v\prime_c v_{u_t}}}-\sum\limits_{(u_t,c)\in D_{neg}}\log\frac{1}{1+e^{v\prime_c v_{u_t}}}-\sum\limits_{(u_t,l_t)\in D_{reject}}\log\frac{1}{1+e^{-v\prime_{l_t} v_{u_t}}}$
- 现有的排序模型：
  - 算法: GBDT
  - 特征：listing　features、user　features、query　features和cross　features，和embedding features
  
  

#### FineTech数据分析

信用卡违约预测
1. 数据加载
2. 数据探索，查看label情况
3. 数据预处理，切分数据集
4. 使用GridSearchCV+Pipeline完成训练与预测
  - 设置分类器：SVM，决策树，RF，KNN
  - 设置分类器超参数：针对不同分类器超参数可能是不同的
  - 设置Pipeline：机器学习的完成流程（数据规范化 Scaler + 机器学习模型训练 Model）
  
信用卡欺诈预测

量化交易


  
  