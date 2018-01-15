import transforms
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

width = 320
height = 160
depth = 3
new_width = 64
new_height = 64


def balance_data(x, y):
    random_transforms = [transforms.brightness, transforms.noise, transforms.shift]
    bins = np.arange(-1, 1.01, 0.1)
    angles_hist, _ = np.histogram(y, bins)
    max_bin = max(angles_hist)
    for steer_bin in range(len(angles_hist)):
        bin_num = angles_hist[steer_bin]
        if 0 < bin_num < max_bin:
            lower_bound = bins[steer_bin]
            upper_bound = bins[steer_bin + 1]
            bin_indexes = np.where((y >= lower_bound) & (y <= upper_bound))[0]
            x_bin, y_bin = [], []
            for i in range(max_bin-bin_num):
                bin_img_index = np.random.choice(bin_indexes)
                transform = np.random.choice(random_transforms)
                new_bin_x, new_bin_y = transform(x[bin_img_index], y[bin_img_index])
                x_bin.append(new_bin_x)
                y_bin.append(new_bin_y)
            x = np.concatenate((x, x_bin))
            y = np.concatenate((y, y_bin))
    return x, y


def load_data():
    trainx_data = '../udacity-data/trainx.npy'
    trainy_data = '../udacity-data/trainy.npy'
    with open(trainx_data, 'rb') as fx, open(trainy_data, 'rb') as fy:
        x_train = np.load(fx)
        y_train = np.load(fy)
    return x_train, y_train


def augment_data(x_train, y_train):
    x_aug, y_aug = [], []
    for img, angle in zip(x_train, y_train):
        img_blur, angle_blur = transforms.blur(img, angle)
        img_gray, angle_gray = transforms.gray(img, angle)
        img_mirror, angle_mirror = transforms.mirror(img, angle)
        x_aug.extend([img_blur, img_gray, img_mirror])
        y_aug.extend([angle_blur, angle_gray, angle_mirror])
    x_train = np.concatenate((x_train, x_aug))
    y_train = np.concatenate((y_train, y_aug))
    return x_train, y_train


def prep_data():
    x_train, y_train = load_data()
    x_train = np.array([cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA) for img in x_train])
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)
    x_train, y_train = augment_data(x_train, y_train)
    x_train, y_train = balance_data(x_train, y_train)
    return x_train, x_validation, y_train, y_validation
