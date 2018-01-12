import cv2
import numpy as np
from matplotlib import pyplot as plt


def mirror(img, angle):
    mirrored_img = cv2.flip(img, 1)
    mirrored_angle = -angle
    return mirrored_img, mirrored_angle


def noise(img, angle, size=64, depth=3):
    noise_max = 20
    noise_mask = np.random.randint(0, noise_max, (size, size, depth), dtype='uint8')
    noisy_img = cv2.add(img, noise_mask)
    return noisy_img, angle


# from Vivek Yadav https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def brightness(img, angle):
    bright_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = 0.25+np.random.uniform()
    bright_img[:, :, 2] = bright_img[:, :, 2] * random_bright
    bright_img = cv2.cvtColor(bright_img, cv2.COLOR_HSV2RGB)
    return bright_img, angle


def blur(img, angle):
    kernel = 3
    blurred_img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blurred_img, angle


def gray(img, angle):
    grayed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grayed_img = cv2.cvtColor(grayed_img, cv2.COLOR_GRAY2RGB)
    return grayed_img, angle


def shift(img, angle, size=64):
    angle_offset_per_pixel = 0.005
    max_shift = 16
    x_shift, y_shift = np.random.random_integers(-max_shift, max_shift, 2)
    shifted_angle = angle + x_shift * angle_offset_per_pixel
    trans_m = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted_img = cv2.warpAffine(img, trans_m, (size, size))

    return shifted_img, shifted_angle


if __name__ == '__main__':
    img_test = plt.imread('images/resized.jpg')
    img_test, img_angle = gray(img_test, 0.5)
    print(img_angle)
    plt.imshow(img_test)
    plt.show()
    plt.imsave('images/out.jpg', img_test)
