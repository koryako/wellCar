11   注意力

     论文《A Differentiable Transition Between Additive and Multiplicative Neurons》对这一概念进行了探索，
     参阅：https://arxiv.org/abs/1604.03736。
     
     
     另外，《深度|深度学习能力的拓展，GoogleBrain讲解注意力模型和增强RNN》这篇文章也对软注意机制进行了很好的概述。 
     http://www.yidianzixun.com/home?page=article&id=0ERwZLZ6
     软注意最简单的形式在图像方面和向量值特征方面并无不同，还是和上面的（1）式一样。

     论文《Show,AttendandTell:NeuralImageCaptionGenerationwithVisualAttention》是最早使用这种类型的注意的研究之一：
     
     https://arxiv.org/abs/1502.03044 


     下面这两个机制解决了这个问题，它们分别是由DRAW（https://arxiv.org/abs/1502.04623）和
     SpatialTransformerNetworks（https://arxiv.org/abs/1506.02025）这两项研究引入的。
     
     它们也可以重新调整输入的大小，从而进一步提升性能。 


     最近一篇关于使用带有注意机制的RNN进行生物启发式目标跟踪的论文HART中就使用了这种机制，参阅：https://arxiv.org/abs/1706.09262。



24

这两个是pytorch版本的；官方https://github.com/LantaoYu/SeqGAN；

seqgan的升级版leakgan


19  动作视频数据集
    1.数据集地址：https://research.google.com/ava/

      论文：https://arxiv.org/abs/1705.08421 
    许多基准数据集，
    UCF101、
    activitynet
    和DeepMind的Kinetics，
    都是采用图像分类的标记方案，