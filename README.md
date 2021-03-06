## baseline: CADA-VAE
基于CADA—VAE实现AAAI2021将解耦用于GZSL的文章。
![image](https://github.com/CallMeNumber1/DR-GZSL/blob/master/model.png)
#### 实现细节
- 目前shuffle classification loss的权重因子设为5，学习率为0.01，然后训练解耦VAE。
- 训练好VAE后，再训练最终分类器，最高可达0.535
- 训练最终分类器时，GZSL上的精度变化：
-   开始时，大概前15个epoch，unseen acc和seen acc会一起提高，
-   随着训练的进行，unseen acc降低，seen acc提高。
#### 一些需要调参的地方
1. warm up阶段对损失权重因子的调整。这个比较难调整，先不调。
> alpha, beta这些权重因子超参数，文章都没给出
2. 辅助分类器，输出可见类数量还是所有类别数量，目前是输出所有类别数量
3. 训练轮次
-   VAE训练轮次，尝试了100，150，200，最后算到200
-   GZSL分类器训练轮次，尝试了23、40、60，最后设为60，事实上到30左右时H值就差不多达到最高了
4. 不同权重因子的变化策略。
-   尝试了1，5，10，目前shuffle classification的factor设置为10效果最好
5. CADA中，测试集数据，reparameter=False，不知为何
#### 关于训练VAE的时候损失最后仍然较大的问题
- 其实能稳定到一个点即可，即能达到稳定状态。因为最后并不是用VAE做分类，而是要另外训练一个分类器。
> 另外，VAE中的重构损失项具有较大的损失。
- CADA-VAE训练100轮后，损失为13000左右，训练200轮，仍然在13000左右。
