import numpy as np
from PIL import ImageGrad
import cv2
import pyautogui
from directkeys import PressKey, W, A, S, D

for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
def main():
    last_time=time.time()
    while(True):
        PressKey(W)
        printscreen=np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('lop took{}seconds'.format(time.time()-last_time))
        last_time=time.time()
        newScreen=edgesDetection(printscreen)
        cv2.imshow('window', newScreen)
        #cv2.imshow('window',cv2,cvtColor(newScreen,cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xff==ord('q'):
             cv2.destroyAllWindows()
             break

def edgesDetection(image):
    original_image=image 
    processed_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    processed_img=cv2.Canny(processed_img,threshold1=200,threshold2=300)

    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]], np.int32)
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    processed_img = roi(processed_img, [vertices])
    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                          edges       rho   theta   thresh         # min length, max gap:        
    lines = cv2.HoughLinesP(processed_img,1,np.pi/180,180,20,15)
    draw_lines(processed_img,lines)
    return processed_img



def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img,lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)

if __name__ == '__main__':
    main()