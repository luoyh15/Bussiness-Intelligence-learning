#### 资金流入流出预测

ARIMA
- 模型过于简单
- 没有考虑weekday的特征规律
- 没有考虑节假日信息


基于周期因子的时间序列预测
1. 计算周期因子（weekday）:也就是周几的权重
2. 计算每日（1号-30号）均值：作为base来进行下个月的预测（但需要把weekday影响去掉）
3. 统计星期几（weekday）在每日（day）出现的频次：确定每日的weekday权重
4. 基于周期因子获得加权均值，得到每日的base：去掉周期因子的影响
5. 根据每日的base和周期因子进行预测：每日base乘上weekday权重


新闻自动化处理
- 提取人物和地点
- 关键词
- 关键句
- 词云可视化

1. 数据清洗
  - 正则表达式
    - python： re
    - 匹配：search match compile findall sub
2. 分词，关键词提取
  - TF-IDF计算
3. 词云展示: WordCloud
4. 自动摘要:TextRank




