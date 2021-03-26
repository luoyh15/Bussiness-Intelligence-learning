### Seq2Seq

- 结构：Encoder+Decoder
- Teacher Forcing
  - Decoder每次解码作为下次的输入，导致错误累积
  - 将输出的label作为输入，加快模型收敛
- 基于Attention
  - 加权重：$c_i=\sum_{j=1}^T a_{ij}h_j$
  - $a_{ij}=softmax(e(s_{i-1}, h_j))$
  - $e$函数：距离/相似度
  
- Beam Search: 同时考虑B个概率最大的结果


### 数据分析思维
- 数据分析
  - 目标（商业）
  - RoadMap：细化目标、方法论
    - 人货场模型   在线销售
    - AIPL模型（Acknowledge、Interest、Purchase、Loyalty）用户运营
    - Kraljic采购定位模型
    - 帕累托法则（二八定律）重要性划分
    - 漏斗分析
    - RFM（Recency，Frequency，Mount）用户分层模型



### project：供应链采购中的BI分析

