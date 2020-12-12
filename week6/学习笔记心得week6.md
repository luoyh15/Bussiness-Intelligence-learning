### 因子分解机Factorization Machine

矩阵分解的局限性：
- 只有user、item两个维度

特征向量：$X=(x_1, x_2,..., x_n)^T$    

预测：
- 引入两两特征交叉组成新特征：$\hat{y}(x)=w_0+\sum\limits_{i=0}^n w_i x_i+\sum\limits_{i,j=0}^n w_{ij} x_i x_j$
- 权重$w_{ij}$是稀疏的，需要进行矩阵分解：
  - $\hat{y}(x)=w_0+\sum\limits_{i=0}^n w_i x_i+\sum\limits_{i,j=0}^n \left<V_i,V_j\right> x_i x_j$
  - $\left<V_i,V_j\right> = \sum\limits_{f=1}^k v_{i,f}\cdot v_{j,f}$ 
  
MF是FM的特例（n=2）
高阶（>2)特征一般不采用，工程上复杂度太高  
工具：libFM

FFM（Field-aware Factorization Machine）
- 每个特征有多个隐向量，交叉时使用对应其他特征的隐向量
- 计算量较大，$kn^2$
- FM是FFM的特例


DeepFM算法：DNN+FM
- 既考虑低阶（1阶+2阶，用FM），又考虑高阶（DNN）
- 推荐系统DNN层数大概3层，每层神经元200-400。（from DeepFM的论文）
- 工具DeepCTR


### 基于领域的协同过滤
基于邻域推荐：KNN  

基于用户的协同过滤（UserCF）：
1. 找到与目标用户相似的用户集合
  - 相似度：Jaccard、cos
2. 取topK个用户对于物品i的兴趣的加权平均值，权重就是相似度
3. 排序进行推荐。（去重）

基于物品的协同过滤（ItemCF）：
1. 计算物品之间的相似度
2. 用户对物品i的兴趣=用户对物品i的k个相似物品的兴趣的的加权平均值
3. 排序进行推荐。（去重）  

基于邻域推荐：
- 用交叉验证来确定K（20-50）
- 冷启动，需要借助其他信息来找到领域K
- UserCF：用相似用户来推荐
- ItemCF：用相似物品来推荐（可解释性强）
- 物品迭代快，用UserCF更准确；物品相对稳定，用ItemCF  

工具：surprise

