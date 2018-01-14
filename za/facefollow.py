# -*- coding: utf-8 -*-
＃＃转载请注明：@小五义http://www.cnblogs.com/xiaowuyi  QQ群：64770604
import cv2.cv as cv  
import cv2  
from cv2 import VideoCapture  
      
#cv.NamedWindow("W1", cv.CV_WINDOW_AUTOSIZE)  
cv.NamedWindow("W1",cv.CV_WINDOW_NORMAL)

cv.ResizeWindow("W1", 600, 600)
      
    #找到设备对象  
capture = cv.CaptureFromCAM(0) 

      
    #检测人脸函数  
      
def repeat():  
      
        #每次从摄像头获取一张图片  
    frame = cv.QueryFrame(capture)

    image_size = cv.GetSize(frame)#获取图片的大小  
    #print image_size
          
    greyscale = cv.CreateImage(image_size, 8, 1)#建立一个相同大小的灰度图像

    cv.CvtColor(frame, greyscale, cv.CV_BGR2GRAY)#将获取的彩色图像，转换成灰度图像

    storage = cv.CreateMemStorage(0)#创建一个内存空间，人脸检测是要利用，具体作用不清楚
          
    cv.EqualizeHist(greyscale, greyscale)#将灰度图像直方图均衡化，貌似可以使灰度图像信息量减少，加快检测速度  
    
    #画图像分割线
         
    cv.Line(frame, (210,0),(210,480), (0,255,255),1) 
    cv.Line(frame, (420,0),(420,480), (0,255,255),1) 
    cv.Line(frame, (0,160),(640,160), (0,255,255),1) 
    cv.Line(frame, (0,320),(640,320), (0,255,255),1) 
        # detect objects  
    cascade = cv.Load('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
    #加载Intel公司的训练库  
      
        #检测图片中的人脸，并返回一个包含了人脸信息的对象faces  
    faces = cv.HaarDetectObjects(greyscale, cascade, storage, 1.2, 2,
                                 cv.CV_HAAR_DO_CANNY_PRUNING,
                                 (100, 100))  
      
        #获得人脸所在位置的数据  
    for (x,y,w,h) , n in faces:
       # print x,y
        if x<210:
            print "right"
        elif x>310:
            print "left"
        cv.Rectangle(frame, (x,y), (x+w,y+h), (0,128,0),2)#在相应位置标识一个矩形 边框属性(0,0,255)红色 20宽度
          
        cv.ShowImage("W1", greyscale)#显示互有边框的图片
          
    cv.ShowImage("W1", frame)  
      
    #循环检测每一帧的图片 ESC键退出程序  
while True:  
    repeat()  
    c = cv.WaitKey(10)  
    if c == 27:  
        #cv2.VideoCapture(0).release()  
        cv2.destroyWindow("W1")  
        break
