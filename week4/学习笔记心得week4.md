### ALS（Alternating Least Squares）矩阵分解算法
- 隐语义模型
  - 用户与物品之间隐含的联系
  - 隐含特征（Latent Factor），隐特征少（粒度粗）；隐特征多（粒度细）
 
 
- 基于邻域的推荐
  - userCF，itemCF
  - 吃内存（memory-based）
  
  
- 推荐系统应用场景
  1. 评分预测。稀疏矩阵，矩阵分解主要应用场景
  2. Top-N推荐。排序模型


- 矩阵分解
  - 用两个低维矩阵（embedding）来表示原来的高维矩阵。实际上是一种降维
  - loss函数来评估分解的合理性。注意用户没有做出过评价的未知的，不能当作零计入loss。
    - $r_{ui}$:用户u对商品i的评分。
    - $x_u$:用户u的向量。$X=[x_1, x_2, ..., x_N]$
    - $y_i$:商品i的向量。 $Y=[y_1, y_2, ..., x_M]$
    - $loss = min\sum\limits_{r_{ui}\neq 0}(r_{ui}-x_u^Ty_i)^2+\lambda(\sum\limits_u||x_u||_2^2+\sum\limits_i||y_i||_2^2)$
  
  
- ALS（Alternating Least Squares）交替最小二乘
  - 优化方法：最小二乘
    - $loss = min\sum\limits_{r_{ui}\neq 0}(r_{ui}-x_u^Ty_i)^2+\lambda(\sum\limits_u||x_u||_2^2+\sum\limits_i||y_i||_2^2)$
    - 此loss适用于显式反馈（用户对商品有评分$r_{ui}$）
    1. 固定Y优化X：$x_u=(YY^T+\lambda I)^{-1}YR^T_u$
    2. 固定X优化Y: $y_i=(XX^T+\lambda I)^{-1}XR_i$
  - 置信度：$c_{ui}=1+\alpha r_{ui}$
    -  $loss = min\sum\limits_{r_ui\neq 0}c_{ui}(p_{ui}-x_u^Ty_i)^2+\lambda(\sum\limits_u||x_u||_2^2+\sum\limits_i||y_i||_2^2)$
    - 此loss适用于隐式反馈（用户对商品没有显式反馈，而是点击次数、观看时长等）
    1. 固定Y优化X：$x_u=(Y\Lambda_u Y^T+\lambda I)^{-1}Y\Lambda_u P^T_u$
    2. 固定X优化Y: $y_i=(X\Lambda_i X^T+\lambda I)^{-1}X\Lambda_i P_i$
  - spark: 大规模数据处理的统一分析引擎,分布式。
    - mllib库、ml库（功能更全面）
    - python代码als.py
    - Hadoop:存储系统（HDFS）
  

- GD(gradient descent)
  - batch GD
  - stochastic GD
  - min-batch GD折中方法


- K-fold：对数据划分成K份，其中一份作为测试集，其他作为训练集，训练K个模型进行交叉验证。


- Baseline算法：基于统计的基准预测线打分
  - $b_{ui}$: 预测值
  - $b_u$: 用户对整体的偏差
  - $b_i$: 商品对整体的偏差
  - 用ALS进行优化


- Surprise（推荐系统工具）常用算法：
  - Baseline算法
  - 基于邻域的协同过滤
  - 矩阵分解：SVD，SVD++，PMF，NMF
  - SlopeOne 协同过滤算法
  
  1. Normal Perdictor：
    - 用户对物品的评分是服从正态分布
    - 根据已有的评分的均值和方差 预测当前用户对其他物品评分的分数
  2. Baseline
    - 设立基线
    - 引入用户的偏差以及item的偏差
  3. SlopeOne
    1. 根据item之间评分差的均值
    2. 用用户评分过的item来推断未评分的item的评分。
    3. 进行排序，返回topN
    - 加权SlopeOne
  4. KNN:
    - K个最近邻居（用户or商品）的加权平均。
    - 用相似度衡量距离
    - KNNBasic, KNNWithMeans, KNNWithZscore, KNNBaseline
  5. 矩阵分解
    - 降维
    - SVD, SVDpp, Non-Negative Matrix-Factorization
   

### Pandas使用
- apply函数
  - apply（DF的row、column）、applymap（DF的element）、map（Series的element）
- 统计函数。describe, count, mean, median,..., var, std, argmax, argmin, idxmax, idxmin
- 表连接merge。类似于数据库SQL操作
- loc, iloc
- groupby

   
    



