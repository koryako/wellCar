import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
csv_path = '../../../run1/driving_log.csv'  # my data
csv_path1 = 'data/driving_log.csv'  # udacity data
 
center_db, left_db, right_db, steer_db = [], [], [], []
Rows, Cols = 64, 64
offset = 0.22

# read csv file
with open(csv_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        center_db.append(row['center'])#center
        left_db.append(row['left'])#left
        right_db.append(row['right'])#right
        steer_db.append(float(row['steering']))
        """
        if float(row['steering']) != 0.0:
            center_db.append(row['center'])
            left_db.append(row['left'])
            right_db.append(row['right'])
            steer_db.append(float(row['steering']))
        else:
            prob = np.random.uniform()
            if prob <= 0.1:
                center_db.append(row['center'])
                left_db.append(row['left'])
                right_db.append(row['right'])
                steer_db.append(float(row['steering']))
         """
plt.figure(figsize=(10,4))
x = [range(len(steer_db))]
x = np.squeeze(np.asarray(x))
y = np.asarray(steer_db)
plt.xlim(0,8000)
plt.title('data distribution', fontsize=17)
plt.xlabel('frames')
plt.ylabel('steering angle')
plt.plot(x,y, 'g', linewidth=0.4)
plt.show()


plt.hist(steer_db, bins= 50, color= 'orange', linewidth=0.1)
plt.title('angle histogram', fontsize=17)
plt.xlabel('steering angle')
plt.ylabel('counts')
plt.show()


def read_img(input):
    img = cv2.imread(input)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

num = 300



center_img, center_steering = read_img(center_db[num].strip()), steer_db[num]
left_img, left_steering = read_img(left_db[num].strip()), steer_db[num] + offset
right_img, right_steering = read_img(right_db[num].strip()), steer_db[num] - offset


plt.figure(figsize=(12,3))
plt.subplot(1,3,1)
plt.imshow(left_img)
plt.title('Left Image ( ang : ' + str(np.round(left_steering,3)) + ' )')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(center_img)
plt.title('Center Image ( ang : ' + str(np.round(center_steering)) + ' )')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(right_img)
plt.title('Right Image ( ang : ' + str(np.round(right_steering)) + ' )')
plt.axis('off')
plt.show()