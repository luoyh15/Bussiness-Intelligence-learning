### SVD矩阵分解

##### project
- 聚类：K-Means, 怎么选取K？
- 手肘法：
  - 不同K所得到的簇内误差平方和SSE（sum of the squared errors）
  - 在拐点处选取K比较合适。
  

##### SVD
- numpy工具：linalg.eig， scipy.linalg.svd  
- 原理：
  - $AA^T=P\Lambda_1 P^T$
  - $A^TA=Q\Lambda_2 Q^T$
  - $A=P\Lambda Q^T$
  - $\Lambda_1, \Lambda_2$ 只有维度不同，值都是一样的。  
- 应用：
  - 图象压缩  
- 推荐系统中的应用：
  - 降维
  - 预测  
- 传统SVD局限
  - 要求矩阵稠密，元素不能有缺失
  - 填充简单导致噪声大    
- FunkSVD：
  - $M_{m,n}=P^T_{k,m}Q_{k,n}$
  - 最小化损失函数：$\sum_{i,j}=(m_{ij}-q^T_j p_i)^2$
  - 防止过拟合加上正则项
  - SGD进行求解
  - 和ALS一样，只是求解方式不同  
- BiasSVD：
  - 用户商品都有自己的偏好（bias）
  - $\hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u$
  - 优化目标：$\text{arg}\min\limits_{p_i,q_j}\sum_{i,j}(m_{ij}-\mu-b_i-b_j-q^T_j p_i)^2+\text{regularizations}$
  - 用SGD进行求解  
- SVD++
  - 在BasicSVD基础上的改进：考虑用户的隐式反馈
  - 隐式反馈：没有具体的评分，可能是点击、浏览。
  - $\hat{r}_{ui} = \mu + b_u + b_i + q_i^T\left(p_u +|I_u|^{-\frac{1}{2}} \sum_{j \in I_u}y_j\right)$
  - $ y_j:$代表隐式反馈的评分。
  - surprise中的SVD++算法把用户对商品进行评分这个行为作为隐式反馈  
- 工具：
  - surprise.prediction_algorithms.matrix_fatorization  SVD、SVDpp   
- 矩阵分解只考虑user和item特征，主要用于召回。 
- 优点：
  - 能将高维的矩阵映射成两个低维矩阵的乘积，很好地解决了数据稀疏的问题；
  - 具体实现和求解都很简洁，预测的精度也比较好；
  - 模型的可扩展性也非常优秀，其基本思想也能广泛运用于各种场景中。  
- 缺点：
  - 可解释性很差，其隐空间中的维度无法与现实中的概念对应起来；
  - 训练速度慢，不过可以通过离线训练来弥补这个缺点；
  - 实际推荐场景中往往只关心topn结果的准确性，此时考察全局的均方差显然是不准确的。  
  
  
Google Colab
- 加载Google drive中的文件
        from google.colab import drive
        drive.mount('/content/drive')

#### 基于内容的推荐
  - 不需要用户的动态信息
  - 冷启动
  
步骤
1. 基于描述的特征提取
  - N-gram。 分词，列出所有词，计算词频
    - 第n个词的出现与前n-1个词相关
    - unigram, bigram, trigram
  - 得到词向量（TF-IDF）
    - 词频（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率。
      - $\text{tf}_{i,j}=\frac{n_{i,j}}{\sum_{k}n_{k,j}}$
    - 逆向文件频率（inverse document frequency，idf）是一个词语普遍重要性的度量。
      - $\text{idf}_{i}=log\frac{|D|}{1+|\{j:t_i\in d_j\}|}$
    - 某一特定文件内的高词语频率$tf_{i,j}$，以及该词语在整个文件集合中的低文件频率$idf_{i}$，可以产生出高权重的tf-idf。因此，tf-idf倾向于过滤掉常见的词语，保留重要的词语。 
      - $\text{tfidf}_{i,j}=\text{tf}_{i,j}\times \text{idf}_i$
2. 计算相似度
  - cos
3. 进行推荐 
 
Word Embedding
- Embedding
  - 降维
  - 构建新的特征空间
  - word2vec
    - MLP(MultiLayer Perceptron)模型
    - 工具：GenSim
    - 用于商品推荐
      - 商品-》单词；用户对商品的行为序列-》文章

