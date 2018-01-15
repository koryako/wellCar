[![Steery - The Artificial Intelligence driver](http://img.youtube.com/vi/--jKKhA8dys/0.jpg)](https://www.youtube.com/watch?v=--jKKhA8dys)

# Steering a car

自动驾驶课程 [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive).

使用模拟器端对端训练打方向
([Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip),
[MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip),
[Win32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip),
[Win64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip))
首先要自己玩模拟器，收集摄像头图片和此时的方向盘角度数据，然后教神经网络从图像中返回角度数据，这就是我们所知道的监督学习。


# How does it work?

There is nice [Nvidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
where they applied this approach to drive a real car. You may find videos of it on [youtube](https://www.youtube.com/results?search_query=nvidia+bb8). It's pretty amazing to watch.
I use the same but simplified method. Since we are working with images we need
[convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) or CNN which is specialized on image processing.
There are a lot of free tutorials over the internet about CNNs. [For example](https://www.udacity.com/course/deep-learning--ud730).
So I decided to use similar to Nvidia CNN.

![CNN architecture](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/Proj3-CNN.png "CNN architecture")

Why this architecture? So, I used grayscaled and normalized to interval [-0.5, 0.5] images to reduce memory and GPU usage on training. That's why I have 1 channel image as input.
I tried to use minimalistic neural network but it's hard to say that it is tiny network. It has 3 129 152 parameters. 110 368 and 3 018 784 parameters
for CNN and fully connected part respectively. It is big number of parameters and additional experimenting probably may decrease that number. Also I
used fully connected layer with 5 hidden layers to allow fully connected network to have its own lower, middle and high level features as it is
done in CNN. Also I used [dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout) with 0.5 probability on all layers except last 3 
relatively small layers to deal with [overfitting](https://en.wikipedia.org/wiki/Overfitting).

# Training data

For all supervised learning tasks such as our behavioral cloning collecting training data has an extremely high importance. We need to collect dataset which
is correctly represent all possible situations that can emerge while driving. I mean situations where we not only driving a car straight across the street and
making turns left or right. But also situations where we need to recover a car from bad positions on the road. Such as various course deviations. As we mostly
drive a car straight such samples of straight driving will probably dominate the dataset and lead to model overfitting. I decided to record number of datasets with
different behavior and mix them togeather:
- Straight driving with turns
- Strong disturbance recovery
- Medium disturbance recovery
- Light disturbance recovery

I recorded disturbance recovery dataset by randomly making disturbance to right and recovery with left steering. So then I just excluded all right turns from that dataset.
Also I repeated that procedure with left turns. So at the end I had 6 dataset with disturbance recovery. 3 for recovery from right and 3 for recovery from left.
I mixed a 50% strong, 100% of medium and light disturbance and excluded 90% of straight drive samples in result dataset which have 12098 samples.
There is 2 tracks in simulator. I used only track 1 for recording samples. Track 2 will be used to test the model how good it generalizes steering a car.

![Steering distribution in the dataset](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/dataset-steering-distribution.png "Steering distribution in the dataset")

Left picture shows steering distribution in the dataset and right picture shows steering distribution in augmented dataset (read further for detalis)

# Dataset augmentation

12098 samples is not big enough dataset to train a good quality model because of overfitting. That's why we need to augment our dataset with generated samples. Our sample is image and corresponding steering wheel position.
Here is an example of input image.

![Dataset image example](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/dataset-image-example.png "Dataset image example")

So we need to generate samples with images and corresponding steering wheel positions. I decided to use these transformations that leaves steering wheel untouched:
- randomize image brightness (-0.3, 0.3)
- randomly partially occlusion with 30 black 25x25 px squares
- very slightely randomly:
    - rotation 1 degree amplitude
    - shift 2px amplitude
    - scale 0.02 amplitude

And
- flipping the image with corresponding flipping of steering wheel value

Also I disturbed steering wheel value with small normal noise (0 mean, 0.005 standard deviation). You can see resulting steering wheel distribution on the right dataset steering distribution image earlier.
Augmented images looks like this:

![Augmented image example](https://github.com/parilo/steering-a-car-behavioral-cloning/blob/master/augmented-image-example.png "Augmented image example")

# Training

For creating and training the model I used [Keras](https://keras.io/) which has big library of standard neural networks blocks. So training neural networks with Keras is pretty simple and fun :)
For training I used only augmented samples, so model haven't seen one sample twice. That is again for preventing the overfitting. I used Adam optimizer with 1e-4 learning rate and mean squared error as loss function.
I decided to use 112 samples batch size and 30 epochs of 44800 samples. Collected dataset I splitted into 67% train and 33% validation parts. As test dataset I used straight driving samples recorded on track 2.
I saved model on every epoch and selected one from last epochs models that is able to drive track 2. I tried several times to train and noticed that not every time it is possible to select such model. But models are close to drive track 2.
Despite of track 2 haven't been used to record samples and has much sharper turns (and higher complexity as for me). Training model can drive it without seeing a single image from it. That fact was very surprising for me.

How to improve the model? There is different ways. So firstly we need much more sophisticated environment in the simulator. Including many other types of turns and crossings,
road materials, off-road places, weather effects like rain, snow, fog, ice, etc. Also we need other cars, pedestians, bicyclists, motorcycles and many other traffic participants.
And possibly it is good to have function that can instantly place the car in random road situation, so it will possibly way to get more behavior rich dataset. Maybe it is helpful
to have such simulator that can randomly generate whole the environment on request.
Secondly, since we may have multiple images on a car we may use 3d reconstruction to generate samples with changed point of view. Like it is done in Nvidia paper or
more detailed. In this case we may need proper camera calibration parameters and cameras placement geometry.
I think further improvement may be reached with reinfocement learning using the model as pretrained actor in actor-critic approach. With RL model may improve errors in driving that
it learned from behavior cloning dataset and possibly find smarter way of driving.

# How to train and run?

- Download simulator ([Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip),
[MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip),
[Win32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip),
[Win64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip))

- Record your dataset. It is csv file with such records
```
center image path, left image path, right image path, steering, throttle, breakvals, speed
```
only center image path and steering column is required other may contain arbitrary values

- Use model.py or Behavioural-Cloning-clean.ipynb ipython notebook to train your model.
    dont forget to change dataset loading part to specify your dataset location.
    GPU is strongly recommended.

- Select autonomous mode and track in simulator. Run command
```
python drive.py model.json
```
Already trained model is included. You can try it :)

Warning: simulator is not synchronised to drive.py. Make sure that simualtor and drive.py works without lag, use the most lower quality in simulator and GPU for CNN if ot is possible
to avoid lags. Lags can significantly influence driving so it will look like very poor driving quality.

# Conclusion

I found this approach very useful and I think it can be used in other simulators and games such as GTA or TORCS. Or even other type of games maybe.




21
AutoML技术，以及新智元分别报道过的“将好奇心加入AI”和“让AI自动分享学到的经验”，将成为狭义AI迈向AGI和超级智能的基础。 


10
[1] Large-ScaleEvolutionofImageClassifiers,
[2]NeuralArchitectureSearchwithReinforcementLearning


# Project 3 - Behavioral Cloning with Keras

The main goal of this project is developement and training a Deep Network to clone human behavioral during driving a car, thus being able to drive autonomously on a simulator, provided by Udacity. The deep network drives the car autonomously along a track, using the image from cenral camera to predict steering angle at each instant.

## General description.

The simulated car is equipped with three cameras, one on the left side, one in the center and one on the right side of the car, that provide images from these different view points. Simulator supports two modes: training mode and autonomous mode. Training mode allows to use all three cameras for recording dataset, used for model training. In autonomous mode only central camera provides images for steering angle prediction. The training Track 1 has bridge, sharp corners, exits, entries, partially missing lane lines of the road bends. An additional test Track 2 for testing the model, has changing elevations, even sharper turns and bumps. It is thus crucial that the Deep Network not only memorize the first track during training, but able to generalize the unseen data to perform correct prediction of steering angles to drive the Track 2 successfully. The developed model was trained only on the training Track 1 and it successfully drives on the test Track 2. The model was trained on Udacity dataset, which contains recorded one lap of the Track 1.

## Dataset

The model was trained on Udacity dataset, containing 8036 samples. Each sample provides the following data, as you can see from the table below:
<ol>
<li>images from three cameras</li>
<li>corresponding steering angle</li>
<li>throttle</li>
<li>break</li>
<li>speed value</li>
</ol>
![data_records](imgs/driving-log-output.png)

#### Image samples

Below you can see typical images from three cameras in each sample record.

![camera_images](imgs/camera_samples.png)

#### Data distribution

Udacity dataset contains record of one lap with human driving along the middle of the road. The analysis of the dataset shows, that the dataset has huge skew in the ground truth data distribution: the steering angle distribution is strongly biased towards the straight direction. 

![training_data_before_preprocessing](imgs/histogram_udacity_dataset.png)

Without accounting for this bias towards the straight direction, the model will not ever learn what to do if it gets off to the side of the road, and as result, the car usually leaves the tack quickly. One way to counteract this problem is to add record data to the dataset when the car is driving from the side of the road back toward the center line.

## Preparation of training dataset

The model was trained on Udacity dataset, containing 8036 samples. As it mentioned above, the first problem with the source dataset is huge bias towards the zero direction. The second problem is small size of training data for training Deep Network with large capacity. To solve these two problems and to improve generalization property of model to drive on unseen tracks, bootstrapping approach (random sampling with replacement) with data augmentation is used to generate a batch with requested size during the training of the model.  

### Data augmentation
<ol>
<li>Exploting all three cameras </li>
<li>Variation of brightness</li>
<li>Horizontal flipping</li>
<li>Horizontal and vertical shifting</li>
<li>Shifting the bias</li>
<li>Image cropping</li>
</ol>

#### Exploting all three cameras
Randomly, the image from one of the center, left or right cameras is used with correction of steering direction as a training sample. This approach, reported in [NVIDIA paper](https://arxiv.org/pdf/1604.07316v1.pdf), allows the model to learn scenarios recovering during driving from the left or right sides of road to the middle of the road.

#### Variation of brightness
Variation of brightness increases variation in trained data to get the model more robust to different light conditions.

#### Horizontal flipping
Horizontal flipping increases variation in trained data and allows to learn the model scenarious to drive along left or right sides of the road with further smooth recovering to the middle of the road. Small vertical flipping is used to increase variation in trained data too with some robustness to horizontal variations.

#### Shifting the bias
The bias parameter with values in range [0, 1] and with random uniform thresholding was added for steering angles to tune the probability of dropout samples with steering angle close to zero from generated trained batch. The effect of bias parameter is demonstrated below on histogram of generated train batch from 2048 samples.
![training_data_after_preprocessing](imgs/histogram_data_steering_angles.png)

#### Image cropping
The preprocessed image is cropped to the size 160x80 of input layer used in the model. The bonnet of the car and half part of sky is removed during cropping too.

## Model architecture

To predict steering angles, a CNN with 730,033 parameters was developed: 3 sequentially connected convolutional layers, 3 full connected layers and one fully connected neuron. The CNN architecture is presented in the table below. The input RGB image is resized to the size of input layer: 160x80, RGB channels. The image normalization to the range [-0.5, 0.5] is implemented in the model as a lambda layer. The output neuron regresses the correct steering value from the features it receives from the previous layers. All three convolutional layers use 3x3 filter with stride equal to 1. The choice of using ELU activation function, instead of more traditional ReLU, come from model of [CommaAI](https://github.com/commaai/research/blob/master/train_steering_model.py), which was developed for the same task of steering regression.

To prevent overfitting of model, three dropout layes were added with drop probability 0.5.


|     Layer (type)     |    Output Shape     | Param #                        
|----------------------|---------------------|-----------
|  Lambda_1 (Lambda)   | (None, 80, 160, 3)  |   0                     
|  Convolution2D       | (None, 78, 158, 16) |   448         
|  ELU (Activation)    | (None, 78, 158, 16) |   0           
|  Maxpooling (2x2)    | (None, 39, 79, 16)  |   0           
|  Convolution2D       | (None, 37, 77, 32)  |   4640        
|  ELU (Activation)    | (None, 37, 77, 32)  |   0           
|  Maxpooling (2x2)    | (None, 12, 25, 32)  |   0           
|  Convolution2D       | (None, 10, 23, 48)  |   13872       
|  ELU (Activation)    | (None, 10, 23, 48)  |   0           
|  Maxpooling (2x2)    | (None, 5, 11, 48)   |   0           
|  Flatten             | (None, 2640)        |   0           
|  Dropout             | (None, 2640)        |   0           
|  Dense               | (None, 256)         |   676096      
|  ELU (Activation)    | (None, 256)         |   0           
|  Dropout             | (None, 256)         |   0           
|  Dense               | (None, 128)         |   32896       
|  ELU (Activation)    | (None, 128)         |   0           
|  Dense               | (None, 16)          |   2064        
|  ELU (Activation)    | (None, 16)          |   0           
|  Dense               | (None, 1)           |   17                             
||||
|Total params: 730,033


## Training the model

The model was compiled with Adam optimizer with default parameters and specified learning rate, equal to 0.001. The model was trained on desktop computer with 4 cores (8 logical cores) on Windows 10.

The bias parameter was fixed as 0.8 but this parameter can be changed along epochs to improve learning performance. The initial dataset is splitted on training and validation sets as 80% vs 20%. The model was trained on 10 epochs, every epoch has 20224 generated samples. Every batch with 256 samples is generated from training dataset with data  augmentation, as described above. 


## Validation the model

The model was evaluated on validation set, 20% of Udacity dataset. Only central camera without any data augmentation is used during generation of validation set.

## Testing the model and Results

After the training the model, when the car smoothly drives along training Track 1, the network can successfully drives  along unseen testing Track 2 too.

A video of the test track performance is shown below.

[![ScreenShot](imgs/track1.jpg)](https://youtu.be/YnP_kDSxEf8)

The performance of the same CNN on the training track is shown below.

[![ScreenShot](imgs/track2.jpg)](https://youtu.be/HE_y7rX2Izo)


## Conclusions

Data augmentation, with according steering angle updates to generate samples for different learning scenarious, allows to train a neural network to recover the car from extreme events, like different row conditions, different lighting conditions, by just simulating such events from regular driving data.

The developed deep network can be improved and redesigned to the aim of improvement generalization of model to drive in more difficult conditions. One of the way to experiment with construction of model, is to use RNN architecture with elements of VGG16 for features extraction to predict next element in context of previous element.



# Project 3: Behavioral Cloning
#### David del Río Medina

### 1. The dataset

The data provided by Udacity was used for this project. To make the most of it, all the images from the three cameras (left, front, right) were used to train an validate the final model. For the images from the left and right cameras, the corrected steering angles must be added to the dataset. Following some other students suggestions (e.g. [An augmentation based deep neural network approach to learn human driving behavior](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)), 0.25 is added (left camera) or subtracted (right camera) to the steering angle defined for the input received by the front camera. Other students have tried different offset values (like 0.27 or 0.3) with success.

After this, the images are resized to 64x64, since it makes training faster and consume less memory.
The data is then split 80/20 in training and validation datasets. Although this can help spot problems with the model, like overfitting, the only valid test is watching the model drive the car through track 1.
The training dataset is not balanced: some steering angles, like 0.0, are overrepresented, which may cause the model to be biased towards driving the car straight. Another problem is that all the data comes from the same track. Even if a good performance in a different setting is not required to pass the project, it would be desirable that the model can generalize to other tracks.

In order to improve generalization, the training dataset is balanced and augmented by applying filters to create new artificial images from the original data.
Six image transformation processes were used:

- Mirror: the image is flipped horizontally and its corresponding steering angle reversed. This is an easy choice, since it is a quick and robust way of having twice the original data.
- Noise: a random value between [0, 20] is added to every color channel of every pixel. This may help the model be less biased towards specific colors, that can change based on the weather, location, time of the day, and such.
- Brightness: the brightness of the original image is changed randomly. The original data is taken at the same hour, with the same weather. This transformation can help the model generalize better under different lightning conditions.
- Blur: Gaussian blur with a 3x3 kernel is applied to the image. The idea is to make the detection of borders and shapes more flexible.
- Gray: the image is converted to grayscale and then back to RGB, to keep the (64, 64, 3) shape. Like the noise transformation, this may help the model be less biased towards specific colors.
- Shift: the original image is shifted horizontally and vertically a random pixel offset between [-16, 16]. The original steering angle is adjusted adding the horizontal offset times 0.005.
This "angle offset per pixel" constant was devised partly by trial and error, partly by estimating the distance in pixels from an image created by the center camera and the corresponding image created by the left camera, following a reference point. This gives an approximate distance of 210 pixels. With the angle correction of 0.25 (also an approximation), we have a correction of 0.25 / 210 = 0.001190476 per pixel for the original sized images, that are 5 times bigger than the resized ones. The estimation suggest a correction of 0.001190476 * 5 = 0.005952381 per pixel for the resized images, but I found out through testing that a constant of 0.005 works better.
This transformation helps generating more data, adding angles that are not present in the original data, and simulating slopes.

###### Some examples:

![Original](images/original.jpg)

_Original image_

![Resized](images/resized.jpg)
![Blur](images/blur.jpg)
![Brightness](images/brightness.jpg)
![Gray](images/gray.jpg)
![Mirror](images/mirror.jpg)
![Noise](images/noise.jpg)
![Shift](images/shift.jpg)

_From left to right: resized to 64x64, blur, brightness, gray, mirror, noise, shift._

The final training dataset is generated following the next procedure:

1. The original data is loaded, including the left and right camera images with their corresponding corrected steering angles.
2. All the images are resized to 64x64.
3. The data is split in training and validation datasets.
4. The training data is augmented by adding three versions of each original image by applying blur, gray and mirror transforms.
5. The training data is balanced by adding transformed versions (noise, brightness and shift) to random images with underrepresented steering angles. Since steering angles are continuous, they are grouped into bins of 0.1 width.


### 2. The architecture

The final architecture follows the diagram below:

![Network diagram](images/netDiagram.png)

The first block consists on 8 convolutional 7x7 filters, followed by a 2x2 max-pooling layer, an activation layer that uses exponential linear units (ELUs) and a batch normalization layer. The choice of ELUs over reLUs is due to the former giving a smoother output. Batch normalization is used in most blocks to keep the values whitened during the whole pipeline.

The second and last block of convolutions is similar to the first one, except this one has 5x5 filters, and dropout of 0.5 is added at the end to prevent overfitting. The output is flattened after this second block.

After the convolutions, there are three blocks of fully connected layers of 512, 256 and 128. Each layer is followed by batch normalization and a ELU activation layer.

The last layer consists on a single neuron that outputs the predicted steering angle, followed by a tanh activation function, that smooths the output and keeps it between -1 and 1 (thanks to [https://carnd-forums.udacity.com/questions/35229158/answers/35229284](https://carnd-forums.udacity.com/questions/35229158/answers/35229284 "Mikel Bober-Irizar")).


### 3. The training

The model was trained using the Adam optimizer and the mean squared error as the objective function to minimize. A small batch size of 32 was used to reduce memory usage. The model was trained for 5 epochs, since a bigger number did not seem to improve the results. All the samples were used in each epoch.


### 4. The results

Using the proposed model for predicting the steering angles, the car is able to drive several times through track 1 without any problems, although its movements are wavy at times:

[![Driving through track one](http://img.youtube.com/vi/lVRYG3_I8HM/0.jpg)](http://www.youtube.com/watch?v=lVRYG3_I8HM)

The car is able to drive through most parts of track 2, but it is incapable to pass one of the steepest slopes, apparently due to lack of speed. This shows that the model is capable of generalizing to unseen circumstances, but a smoother driving (less wavy) would probably allow the car to speed up and work better in track 2.
