课程大纲第一课：SLAM概论和架构
1.从机器人的体系结构讨论SLAM的提出和发展
2.滤波器是什么，谁真正的推动了SLAM？
3.SLAM的新突破-图优化
4.SLAM的完整知识体系结构介绍，基于Linux和ROS进行SLAM的进行本课程学习
5.ROS基础：RGB-D点云示例

第二课：SLAM基本理论一：坐标系、刚体运动和李群
1.SLAM的数学表达
2.欧式坐标系和刚体姿态表示
3.李群和李代数
4.实例：Eigen和Sophus在滤波器上的应用

第三课：SLAM基本理论二：从贝叶斯开始学滤波器

1.随机状态和估计
2.卡尔曼滤波器
3.扩展卡尔曼滤波器和SLAM
4.粒子滤波器和SLAM

5.实例：基于卡尔曼滤波器的SLAM实例

第四课：SLAM基本理论三：图优化

1.从滤波器的痛来谈图优化
2.CovisibilityGraph和最小二乘

3.浅谈Marginlization
4.实例：G2O图优化实战

第五课：SLAM的传感器
1.SLAM传感器综述
2.视觉类传感器（单目、双目和RGBD相机）a.相机模型和标定b.特征提取和匹配
3.主动类传感器--激光a.激光模型和不同激光特性b.激光特征和匹配
4.实例：a.特征提取和立体视觉的深度结算；b.激光数据的基本处理

第六课：视觉里程计和回路检测
1.视觉里程计的综述
2.基于特征法的视觉里程计：PNP
3.基于直接法的视觉里程计：PhotometricError
4.基于立体视觉法的:ICP
5.基于词袋模型的回路检测
6.实例：a.PNP位姿估计b.直接法位姿估计c.回路检测

第七课：激光里程计和回路检测
1.激光里程计简介
2.激光里程计算法LOAM和VLOAM简单介绍
3.激光回路检测的特殊性和主要难点
4.伯克利的BLAM和谷歌Cartographer中回路检测的核心思路介绍
5.实例:LOAM,Cartographer测试

第八课：地图以及无人驾驶系统
1.SLAM中的不同地图系统介绍
2.高精度地图介绍
3.语义地图介绍
4.拓扑地图介绍
5.实例：粒子滤波定位实现

第九课：视觉和无人机、室内辅助导航和AR/VR

1.视觉SLAM的整体重述和实战

2.SLAM、无人机和状态机

3.GoogleTango和盲人导航
4.SLAM的小刺激：AR/VR
5.实例：视觉SLAM的AR实例

第十课：深度学习和SLAM

1.SLAM的过去、现在和未来
2.长航程SLAM的可能性
3.单目深度估计和分割和场景语义
4.动态避障
5.新的特征表达
6.课程总结 

5    
     大数据自动驾驶 
     End-to-end Learning of Driving Models from Large-scale Video Datasets
     https://arxiv.org/abs/1612.01079


记忆 注意力 与 语义 



使用spark 运行ros 运行模拟

测试数据集
www.cvlibs.net/datasets/kitti/

knowlege of   MASK-RCNN traffic segment reconignzs
原文：https://news.voyage.auto/under-the-hood-of-a-self-driving-car-78e8bbce62a6

ROS：http://www.ros.org/

FORScan：http://www.forscan.org/Dataspeed

线控套装：http://dataspeedinc.com/ 

http://blog.csdn.net/AdamShan/article/details/78248421?locationNum=7&fps=1

http://blog.csdn.net/AdamShan/article/details/78265754?locationNum=2&fps=1


https://github.com/udacity/CarND-Extended-Kalman-Filter-Project/tree/master/src

http://blog.csdn.net/lybaihu/article/details/54943545?locationNum=8&fps=1
  贝叶斯对抗生成网络论文地址：https://arxiv.org/pdf/1705.09558.pdf
https://github.com/davidbrai/deep-learning-traffic-lights  交通灯识别
https://github.com/awjuliani
https://github.com/awjuliani/TF-Tutorials

http://blog.csdn.net/weixin_37239947/article/details/74939650处理

tensor2tensor
http://blog.csdn.net/shenxiaolu1984/article/details/73736259 
http://blog.csdn.net/amds123/article/details/73485914
https://github.com/tensorflow/tensor2tensor

生成对抗文本到图像的合成（Generative Adversarial Text to Image Synthesis）
　　【代码】https://github.com/paarthneekhara/text-to-image


http://www.ctoutiao.com/172487.html gan 代码


https://github.com/priya-dwivedi  udacity 学生代码

http://lib.csdn.net/article/aiframework/60538

http://lib.csdn.net/article/89/61236?knId=1818


