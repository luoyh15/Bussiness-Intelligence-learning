# 用户画像 
传统机器学习VS深度模型
- 传统机器学习：人工特征提取，可解释性强，数据需求量相对不大
- 深度模型：从基础特征构建多层复杂特征，可解释性差，数据量需求大

推荐算法:
  - 基于标签
  - 基于内容
  - 基于协同过滤
  - CTR预估
  
## 用户画像
1. 是谁：统一标识。同一体系下，不同APP之间同一个用户的信息共享。
2. 从哪来： 打标签。用户、消费、行为、分析
3. 到哪去：指导业务。

用户生命周期：
- 获客：精准营销
- 粘客：个性化推荐
- 留客：流失率预测

数据-》算法-》业务

聚类算法
- KMeans：分为K类O(n.m.k)
  1. 随机选取K个点作为初始类中心（初始选择对聚类结果和运行时间有很大的影响）
  2. 将每个点分配到最近的类中心，重新计算每个类中心
  3. 重复2，直到类不再变化或者设置最大迭代次数
  - KMeans++:在初始选取第i+1个类中心时，选取离前i个类中心距离最远的。
  - ISODATA（迭代自组织数据分析法）：K值不需人工设定。当属于某个类别的样本数过少时把这个类别去除，当属于某个类别的样本数过多、分散程度较大时把这个类别分为两个子类别。
  - Kernel K-means：参照支持向量机中核函数的思想，将所有样本映射到另外一个特征空间中再进行聚类。
  - 二分k-means：属于层次聚类，采用自顶向下的思想。先进行k=2的k-means聚类，再选择误差平方和较大的重复前一步骤。
  
- faiss, lsh

- EM聚类：Expectation and Maximization

- 层次聚类：聚合（自底向上）or 分裂（自顶向下）

- KNN(K-Nearest Neighbors): 依据最相邻的k个对象中占优的类别进行决策。属于<font color='#dd0000'>分类算法</font>

数据规范化：
- Min-Max
- Z-score


### 基于标签
+ 标签：
  - 可以理解为embedding，做量化
  - User=>Item之间的匹配程度
  - 专家生产or用户生产，
  - 高维事物的抽象（降维）：聚类算法（K-Means，EM聚类，Mean-Shift，DBSCAN，层次聚类，PCA）
- SimpleTagBased
  - 用户的常用标签
  - 商品的标签
  - 用户u对商品i的兴趣：$score(u,i) = \sum_i user_{tages}[u,t]*tag_{items}[t,i]$
- NormTagBased:
  - 简单归一化：$score(u,i) =\sum_i\frac{user_{tages}[u,t]}{user_{tages}[u]}*\frac{tag_{items}[t,i]}{tag_{items}[t]}$
- TagBased-TFIDF：
  - 词频(TF)\逆向文档频率(IDF)
  
  
评价指标：混淆矩阵
- 准确率：$accuracy=\frac{TP+TN}{TP+FP+TN+FN}$
- 召回率：$recall = \frac{TP}{TP+FN}$
- 精确率：$precision=\frac{TP}{TP+FP}$
- F值：$F=\frac{(\alpha^2+1)percision*recall}{\alpha^2(percision+recall)}$


## project相关
AutoML：自动机器学习
- TPOT：
  - 基于Python的AutoML
  - 采用遗传算法选择最优模型及参数
- Google AutoML：
  - 深度学习


数据清洗
- 完整性：数据补全（均值、众数、插值）
- 全面性
- 合法性
- 唯一性
