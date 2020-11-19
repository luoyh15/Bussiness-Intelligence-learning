week2 数据中的关联规则
- 关联规则(衡量参数)
    - 支持度：某件物品或者某组物品出现频率, $supp(A) = 包含A的交易数/总交易数$
    - 置信度：在有A的交易中，出现B的频率, $conf(A\Rightarrow B) = \frac{supp(A\cup B)}{supp(A)}$ 
    - 提升度：在有A的交易中，B的频率与其本身的频率的比值, $lift(A\Rightarrow B) = \frac{conf(A\Rightarrow B)}{supp(B)} = \frac{supp(A\cup B)}{supp(A)supp(B}$
    - leverage: $leve(A\Rightarrow B) = supp(A\Rightarrow B) - supp(A)supp(B)$
    - conviction: $conv(A\Rightarrow B) = \frac{1-supp(B)}{1-conf(A\Rightarrow B)}$


- Apriori算法
    - 一步一步扩大组合内物品数量
    - 设置最小支持度 过滤掉一些组合 得到频繁项集(频繁项集的子集也是频繁项集)
    - 在频繁项集的基础上 设置最小置信度 得到关联规则
    - 工具包：efficent_apriori、mlxtend（需要one_hot）
   
   
- 关联规则步骤
    1. 数据集转换成交易形式
    2. 设置支持度、置信度等参数挖掘关联规则
    3. 按置信度、提升度进行排序，给出推荐结果
    - 场景：BreadBasket、MovieLens、MovieActors


- FPGrowth算法：
    - 构建频繁项集
    - 树形结构
    - 效率比Apriori算法高
    - 工具包 fptools、spark.mllib
    
    
- 相关性分析
    - 为线性回归提供指导依据
    - 相关性系数：
      - Pearson系数:$\rho_{x,y}=\frac{cov(x,y)}{\sigma_x \sigma_y}$
      - kendall: 分类变量相关性
      - spearman：非正态分布数据相关性
    - R（r-square）： 
      - 确定性系数:衡量预测偏差和方差之比
      - $R^2=1-\frac{\sum_{i=1}^{n}(y_i-f(x_i))^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$


- 回归分析
  - 线性回归、多项式回归
  - 损失函数：MSE、MAE
  - 最小二乘：$y=ax+b$
    - 系数计算：$a=\frac{(X-\bar{X})^T(Y-\bar{Y})}{(X-\bar{X}^T(X-\bar{X})}$