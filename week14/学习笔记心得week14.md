## 强化学习

- 机器学习的分支:监督学习\无监督学习\强化学习
  - vs监督学习:没有label数据,只有奖励值
  - vs非监督学习:有奖励值

- 特点:
  - 只有奖励值
  - 奖励不一定是实时的,很可能是延后的
  - 时间是重要因素
  - 当前行为影响后续选择

- 基本概念:
  - Agent
  - Environment
  - Action
  - State
  - Reward
  - Policy
  - Probability
  
- 基本过程:
  - 个体与环境的交互问题
    - 个体: 评估环境\做出行为\得到环境反馈
    - 环境: 接受动作\更新环境\给个体反馈
  - 找到一个最优策略,最大化奖励
    - 所有问题都可以被描述成最大化累积奖励
    - 序列决策:长期\奖励延迟\可能牺牲短期奖励获取更多的长期奖励

- Markov状态:
  - 当前状态\未来状态\之前状态 相互独立
  
  
- Agent分类:
  - value-base:价值
    - 价值网络,基于每个state的评分
    - Q-learning:
      - Q函数:动作-状态对的数值
      1. 初始Q为任意数值
      2. 根据环境奖励更新Q: $Q(s_t, a_t)=(1-\alpha)Q(s_t, a_t)+\alpha(r_t+\gamma\cdot \max_a Q(s_{t+1}, a))$
      - $\alpha$:学习率
      - $R_{t+1}$:奖励
      - $\gamma$:贴现率
  - policy-base:策略
    - 策略网络:给出选择可能性
    - 固定策略的缺陷:状态重名.需要$\epsilon$greedy
    
  - actor-critic:价值+策略


- AlphaGo
  - policy:落子概率
  - 利用MCTS进行策略评估
    - 选择\模拟\拓展\回传
    - UCB算法
  - 神经网络进行评分
  - policy gradient
    - loss=-log(prob)*$v_t$
   
   
   
![Reforce Learning Structure](D:\class\Business Intelligent\week14\RL-structure.png)
 