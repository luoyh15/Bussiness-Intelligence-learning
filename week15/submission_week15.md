**Thinking1：请简述基于蒙特卡洛的强化学习的原理？**

蒙特卡洛是一种随机搜索策略，为当前Action进行评估，评估结果作为神经网络的输入，不断迭代使得神经网络能够给出比较准确的当前State的评估，从而达到强化学习的目的。

**Thinking2：强化学习的目标与深度学习的目标有何区别？**

强化学习的目标是使得Agent能够对当前环境做出最佳的决策。

深度学习属于监督学习，目标是能够学习到数据的一般化规律。



**Action1：训练10$\times$10的五子棋版Alpha GO Zero**

调整棋盘大小，训练1500个batch，发现最佳模型还下不过深度为2000的蒙特卡洛搜索树。人能够轻松赢过AI。

可能原因：

1. 棋盘变大而神经网络结构没变，需要增大CNN的通道数。
2. 带策略价值网络的蒙特卡洛搜索树的深度只有400，需要增大。
3. 每次搜索选择增多，每个状态之间差异较大，需要增大一次训练的batch_size





