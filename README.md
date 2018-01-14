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


<<<<<<< HEAD

https://github.com/priya-dwivedi


Laplace 算子
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html?highlight=laplace


sobel算子
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html?highlight=sobel

Canny
原理 http://www.pclcn.org/study/shownews.php?lang=cn&id=111
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html?highlight=canny#canny

http://selfdrivingcars.mit.edu/resources


三角剖分的算法比较成熟。目前有很多的库（包括命令行的和GUI的可以用）。

常用的算法叫Delaunay Triangulation，具体算法原理见 http://www.cnblogs.com/soroman/archive/2007/05/17/750430.html

这里收集一些开元的做可以测试三角剖分的库
1. Shewchuk的http://www.cs.cmu.edu/~quake/triangle.html，据说效率非常高！
2. MeshLab http://www.cs.cmu.edu/~quake/triangle.html，非常易于上手，只要新建工程，读入三维坐标点，用工具里面的Delaunay Trianglulation来可视化就好了。而且它是开源的！具体教程去网站上找吧。
3. Qhull http://www.qhull.org/
4. PCL库，http://pointclouds.org/documentation/tutorials/greedy_projection.php

无序点云快速三角化

http://www.pclcn.org/study/shownews.php?lang=cn&id=111


opencv 基本操作 https://segmentfault.com/a/1190000003742422

cv 到cv2的不同 http://www.aiuxian.com/article/p-395730.html


已经fork
https://github.com/mbeyeler/opencv-machine-learning

前言

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/00.00-Preface.ipynb

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/00.01-Foreword-by-Ariel-Rokem.ipynb

机器学习的味道

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/01.00-A-Taste-of-Machine-Learning.ipynb

在OpenCV中使用数据

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.00-Working-with-Data-in-OpenCV.ipynb

使用Python的NumPy软件包处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.01-Dealing-with-Data-Using-Python-NumPy.ipynb
在Python中加载外部数据集
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.02-Loading-External-Datasets-in-Python.ipynb
使用Matplotlib可视化数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.03-Visualizing-Data-Using-Matplotlib.ipynb
使用OpenCV的TrainData容器处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.05-Dealing-with-Data-Using-the-OpenCV-TrainData-Container-in-C%2B%2B.ipynb
监督学习的第一步

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.00-First-Steps-in-Supervised-Learning.ipynb

用评分功能测量模型性能
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.01-Measuring-Model-Performance-with-Scoring-Functions.ipynb
了解k-NN算法
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.02-Understanding-the-k-NN-Algorithm.ipynb
使用回归模型预测持续成果
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.03-Using-Regression-Models-to-Predict-Continuous-Outcomes.ipynb
应用拉索和岭回归
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.04-Applying-Lasso-and-Ridge-Regression.ipynb
使用Logistic回归分类虹膜物种
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.05-Classifying-Iris-Species-Using-Logistic-Regression.ipynb
代表数据和工程特性

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.00-Representing-Data-and-Engineering-Features.ipynb

预处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.01-Preprocessing-Data.ipynb
减少数据的维度
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.02-Reducing-the-Dimensionality-of-the-Data.ipynb
代表分类变量
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.03-Representing-Categorical-Variables.ipynb
表示文本特征
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.04-Represening-Text-Features.ipynb
代表图像
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.05-Representing-Images.ipynb
使用决策树进行医学诊断

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb

建立你的第一决策树
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.01-Building-Your-First-Decision-Tree.ipynb
使用决策树诊断乳腺癌
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.02-Using-Decision-Trees-to-Diagnose-Breast-Cancer.ipynb
使用决策树回归
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.03-Using-Decision-Trees-for-Regression.ipynb
用支持向量机检测行人

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.00-Detecting-Pedestrians-with-Support-Vector-Machines.ipynb

实施您的第一支持向量机
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.01-Implementing-Your-First-Support-Vector-Machine.ipynb
检测野外行人
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.02-Detecting-Pedestrians-in-the-Wild.ipynb
附加SVM练习
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.03-Additional-SVM-Exercises.ipynb
用贝叶斯学习实现垃圾邮件过滤器

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.00-Implementing-a-Spam-Filter-with-Bayesian-Learning.ipynb

实现我们的第一个贝叶斯分类器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.01-Implementing-Our-First-Bayesian-Classifier.ipynb
分类电子邮件使用朴素贝叶斯
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.02-Classifying-Emails-Using-Naive-Bayes.ipynb
用无监督学习发现隐藏的结构

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.00-Discovering-Hidden-Structures-with-Unsupervised-Learning.ipynb

了解k均值聚类
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.01-Understanding-k-Means-Clustering.ipynb
使用k-Means压缩彩色图像
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.02-Compressing-Color-Images-Using-k-Means.ipynb
使用k-Means分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.03-Classifying-Handwritten-Digits-Using-k-Means.ipynb
实施聚集层次聚类
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.04-Implementing-Agglomerative-Hierarchical-Clustering.ipynb
使用深度学习分类手写数字

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.00-Using-Deep-Learning-to-Classify-Handwritten-Digits.ipynb

了解感知器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.01-Understanding-Perceptrons.ipynb
在OpenCV中实现多层感知器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.02-Implementing-a-Multi-Layer-Perceptron-in-OpenCV.ipynb
认识深度学习
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.03-Getting-Acquainted-with-Deep-Learning.ipynb
在OpenCV中培训MLP以分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.04-Training-an-MLP-in-OpenCV-to-Classify-Handwritten-Digits.ipynb
训练深层神经网络使用Keras分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.05-Training-a-Deep-Neural-Net-to-Classify-Handwritten-Digits-Using-Keras.ipynb
将不同的算法合并成一个合奏

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.00-Combining-Different-Algorithms-Into-an-Ensemble.ipynb

了解组合方法
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.01-Understanding-Ensemble-Methods.ipynb
将决策树组合成随机森林
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.02-Combining-Decision-Trees-Into-a-Random-Forest.ipynb
使用随机森林进行人脸识别
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.03-Using-Random-Forests-for-Face-Recognition.ipynb
实施AdaBoost
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.04-Implementing-AdaBoost.ipynb
将不同的模型组合成投票分类器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.05-Combining-Different-Models-Into-a-Voting-Classifier.ipynb
使用超参数调整选择正确的模型

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.00-Selecting-the-Right-Model-with-Hyper-Parameter-Tuning.ipynb

评估模型
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.01-Evaluating-a-Model.ipynb
了解交叉验证，Bootstrapping和McNemar的测试
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.02-Understanding-Cross-Validation-Bootstrapping-and-McNemar's-Test.ipynb
使用网格搜索调整超参数
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.03-Tuning-Hyperparameters-with-Grid-Search.ipynb
链接算法一起形成管道
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.04-Chaining-Algorithms-Together-to-Form-a-Pipeline.ipynb
结束语

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/12.00-Wrapping-Up.ipynb
--------------
https://github.com/tensorflow/models/tree/master/object_detection
mobilenet

https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html

带有MobileNets的SSD(Single Shot Multibox Detector)

带有Inception V2的SSD

带有Resnet 101的R-FCN（Region-based Fully Convolutional Networks）

带有Resnet 101的 Faster RCNN

带有Inception Resnet v2的Faster RCNN

https://cloud.google.com/blog/big-data/2017/06/training-an-object-detector-using-cloud-machine-learning-engine



https://github.com/tensorflow/tensorflow/commit/055500bbcea60513c0160d213a10a7055f079312


mobil net 
https://github.com/tensorflow/models/tree/master/inception 准备数据
https://github.com/zehaos/MobileNet

https://github.com/balancap/SSD-Tensorflow


2017.9 

https://github.com/udacity/CarND-Term1-Starter-Kit  环境配置


http://blog.csdn.net/xukai871105/article/details/39255089 树莓派mqtt

https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb
=======

https://github.com/priya-dwivedi


Laplace 算子
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html?highlight=laplace


sobel算子
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html?highlight=sobel

Canny
原理 http://www.pclcn.org/study/shownews.php?lang=cn&id=111
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html?highlight=canny#canny

http://selfdrivingcars.mit.edu/resources


三角剖分的算法比较成熟。目前有很多的库（包括命令行的和GUI的可以用）。

常用的算法叫Delaunay Triangulation，具体算法原理见 http://www.cnblogs.com/soroman/archive/2007/05/17/750430.html

这里收集一些开元的做可以测试三角剖分的库
1. Shewchuk的http://www.cs.cmu.edu/~quake/triangle.html，据说效率非常高！
2. MeshLab http://www.cs.cmu.edu/~quake/triangle.html，非常易于上手，只要新建工程，读入三维坐标点，用工具里面的Delaunay Trianglulation来可视化就好了。而且它是开源的！具体教程去网站上找吧。
3. Qhull http://www.qhull.org/
4. PCL库，http://pointclouds.org/documentation/tutorials/greedy_projection.php

无序点云快速三角化

http://www.pclcn.org/study/shownews.php?lang=cn&id=111


opencv 基本操作 https://segmentfault.com/a/1190000003742422

cv 到cv2的不同 http://www.aiuxian.com/article/p-395730.html


已经fork
https://github.com/mbeyeler/opencv-machine-learning

前言

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/00.00-Preface.ipynb

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/00.01-Foreword-by-Ariel-Rokem.ipynb

机器学习的味道

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/01.00-A-Taste-of-Machine-Learning.ipynb

在OpenCV中使用数据

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.00-Working-with-Data-in-OpenCV.ipynb

使用Python的NumPy软件包处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.01-Dealing-with-Data-Using-Python-NumPy.ipynb
在Python中加载外部数据集
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.02-Loading-External-Datasets-in-Python.ipynb
使用Matplotlib可视化数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.03-Visualizing-Data-Using-Matplotlib.ipynb
使用OpenCV的TrainData容器处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.05-Dealing-with-Data-Using-the-OpenCV-TrainData-Container-in-C%2B%2B.ipynb
监督学习的第一步

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.00-First-Steps-in-Supervised-Learning.ipynb

用评分功能测量模型性能
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.01-Measuring-Model-Performance-with-Scoring-Functions.ipynb
了解k-NN算法
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.02-Understanding-the-k-NN-Algorithm.ipynb
使用回归模型预测持续成果
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.03-Using-Regression-Models-to-Predict-Continuous-Outcomes.ipynb
应用拉索和岭回归
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.04-Applying-Lasso-and-Ridge-Regression.ipynb
使用Logistic回归分类虹膜物种
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.05-Classifying-Iris-Species-Using-Logistic-Regression.ipynb
代表数据和工程特性

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.00-Representing-Data-and-Engineering-Features.ipynb

预处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.01-Preprocessing-Data.ipynb
减少数据的维度
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.02-Reducing-the-Dimensionality-of-the-Data.ipynb
代表分类变量
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.03-Representing-Categorical-Variables.ipynb
表示文本特征
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.04-Represening-Text-Features.ipynb
代表图像
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.05-Representing-Images.ipynb
使用决策树进行医学诊断

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb

建立你的第一决策树
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.01-Building-Your-First-Decision-Tree.ipynb
使用决策树诊断乳腺癌
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.02-Using-Decision-Trees-to-Diagnose-Breast-Cancer.ipynb
使用决策树回归
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.03-Using-Decision-Trees-for-Regression.ipynb
用支持向量机检测行人

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.00-Detecting-Pedestrians-with-Support-Vector-Machines.ipynb

实施您的第一支持向量机
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.01-Implementing-Your-First-Support-Vector-Machine.ipynb
检测野外行人
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.02-Detecting-Pedestrians-in-the-Wild.ipynb
附加SVM练习
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.03-Additional-SVM-Exercises.ipynb
用贝叶斯学习实现垃圾邮件过滤器

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.00-Implementing-a-Spam-Filter-with-Bayesian-Learning.ipynb

实现我们的第一个贝叶斯分类器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.01-Implementing-Our-First-Bayesian-Classifier.ipynb
分类电子邮件使用朴素贝叶斯
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.02-Classifying-Emails-Using-Naive-Bayes.ipynb
用无监督学习发现隐藏的结构

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.00-Discovering-Hidden-Structures-with-Unsupervised-Learning.ipynb

了解k均值聚类
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.01-Understanding-k-Means-Clustering.ipynb
使用k-Means压缩彩色图像
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.02-Compressing-Color-Images-Using-k-Means.ipynb
使用k-Means分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.03-Classifying-Handwritten-Digits-Using-k-Means.ipynb
实施聚集层次聚类
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.04-Implementing-Agglomerative-Hierarchical-Clustering.ipynb
使用深度学习分类手写数字

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.00-Using-Deep-Learning-to-Classify-Handwritten-Digits.ipynb

了解感知器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.01-Understanding-Perceptrons.ipynb
在OpenCV中实现多层感知器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.02-Implementing-a-Multi-Layer-Perceptron-in-OpenCV.ipynb
认识深度学习
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.03-Getting-Acquainted-with-Deep-Learning.ipynb
在OpenCV中培训MLP以分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.04-Training-an-MLP-in-OpenCV-to-Classify-Handwritten-Digits.ipynb
训练深层神经网络使用Keras分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.05-Training-a-Deep-Neural-Net-to-Classify-Handwritten-Digits-Using-Keras.ipynb
将不同的算法合并成一个合奏

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.00-Combining-Different-Algorithms-Into-an-Ensemble.ipynb

了解组合方法
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.01-Understanding-Ensemble-Methods.ipynb
将决策树组合成随机森林
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.02-Combining-Decision-Trees-Into-a-Random-Forest.ipynb
使用随机森林进行人脸识别
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.03-Using-Random-Forests-for-Face-Recognition.ipynb
实施AdaBoost
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.04-Implementing-AdaBoost.ipynb
将不同的模型组合成投票分类器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.05-Combining-Different-Models-Into-a-Voting-Classifier.ipynb
使用超参数调整选择正确的模型

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.00-Selecting-the-Right-Model-with-Hyper-Parameter-Tuning.ipynb

评估模型
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.01-Evaluating-a-Model.ipynb
了解交叉验证，Bootstrapping和McNemar的测试
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.02-Understanding-Cross-Validation-Bootstrapping-and-McNemar's-Test.ipynb
使用网格搜索调整超参数
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.03-Tuning-Hyperparameters-with-Grid-Search.ipynb
链接算法一起形成管道
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.04-Chaining-Algorithms-Together-to-Form-a-Pipeline.ipynb
结束语

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/12.00-Wrapping-Up.ipynb
--------------
https://github.com/tensorflow/models/tree/master/object_detection
mobilenet

https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html

带有MobileNets的SSD(Single Shot Multibox Detector)

带有Inception V2的SSD

带有Resnet 101的R-FCN（Region-based Fully Convolutional Networks）

带有Resnet 101的 Faster RCNN

带有Inception Resnet v2的Faster RCNN

https://cloud.google.com/blog/big-data/2017/06/training-an-object-detector-using-cloud-machine-learning-engine



https://github.com/tensorflow/tensorflow/commit/055500bbcea60513c0160d213a10a7055f079312


mobil net 
https://github.com/tensorflow/models/tree/master/inception 准备数据
https://github.com/zehaos/MobileNet

https://github.com/balancap/SSD-Tensorflow


2017.9 

https://github.com/udacity/CarND-Term1-Starter-Kit  环境配置


http://blog.csdn.net/xukai871105/article/details/39255089 树莓派mqtt

https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb


交通识别问题,行人识别，目标追踪

目标检测
1. 传统方法 
a.2001 paul viola 和Micahel jones 鲁棒实时目标检测 的 viola-jones 框架

b.梯度直方图 Hog 

2.深度学习
a. overFeat 利用卷积 多尺度窗口滑动
b. r-cnn   选择性搜索 selective Search 提取可能目标；使用cnn 在该区域上提取特征；向量机分类
c. fast-rcnn  选择性搜索，cnn 提取特征， 区域兴趣池化 Region of interest ,ROI; 反向传播做分类和边框回归
d. yolo
e. faster-rcnn   cnn 提取特征;regio Proosal network 根据物体的分数来输出可能的目标；区域兴趣池化 Region of interest ,ROI pooling; 反向传播做分类和边框回归
f. SSd  在yolo 上改进，使用了多尺度特征图
g。 R-fcn 使用了 Faster-Rcnn的架构
https://tryolabs.com/blog/
3.数据集
imageNet
coco
Pascal VOC
Oxford-IIIT Pet
kitti Vision


http://www.dev-c.com/nativedb/

github.com/osrf/car_demo

https://github.com/openai/roboschool

gym.openai.com

https://mujoco.org

https://github.com/DartEnv/ddart-env

https://github.com/openai/baselines


目标跟踪

http://www.cs.cityu.edu.hk/~yibisong/iccv17/index.html
convolutional Residual learning for visual tracking


pytorch  caffe2 cntk 之间模型转换用onnx格式
github.com/onnx/onnx

github.com/nottombrown/rl-teacher
https://github.com/nottombrown/rl-teacher.git


https://github.com/aleju/self-driving-truck

https://pan.baidu.com/s/1pL9J4Cz  ros book

https://github.com/qboticslabs/ros_robotics_projects

https://cps-vo.org/group/CATVehicleTestbed/wiki

github.com/tigerneil/deep-reinforcement-learning-family

https://github.com/tigerneil/awesome-deep-rl

https://github.com/facebookresearch/ELF 开源游戏平台

https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md




