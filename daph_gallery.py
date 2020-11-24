import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def create_daphnia_image(ID, clip, ratio_x, ratio_y, seed):
    size = 200
    small_w = 15
    small_h = 15
    epsilon = 0.0001
    t = 0.3
    l = 1.565
    base = 200
    gau_a = 1.0
    gau_c = 12.0
    
    img = np.zeros((size, size), np.uint8)
    
    dx = (2.0 - 2 * epsilon) / (size)
    fx = -1.0 + epsilon
    while fx < 1.0:
        dy = t * ((1.0 + fx)/(1.0 - fx))**((1.0 - l) / (2.0 * (1.0 + l))) * (1.0 - fx * fx)**0.5
        y_top = ratio_y * size * (0.5 - dy)
        y_bottom = ratio_y * size * (0.5 + dy)
        x = ratio_x * (fx / dx + size / 2)
        cv2.line(img, (int(x), int(y_top)), (int(x), int(y_bottom)), float(base), 1)
        fx += ratio_x * dx
    
    img_dt = cv2.distanceTransform(img, distanceType = cv2.DIST_L1, maskSize = 3).astype(np.float32)
    img_l = np.float32(img)
    
    fx = -1.0 + epsilon
    while fx < 1.0:
        dy =  t * ((1.0 + fx)/(1.0 - fx))**((1.0 - l) / (2.0 * (1.0 + l))) * (1.0 - fx * fx)**0.5
        y_top = ratio_y * size * (0.5 - dy)
        y_bottom = ratio_y * size * (0.5 + dy)
        x = ratio_x * (fx / dx + size / 2)
        cv2.line(img_l, (int(x), int(y_top)), (int(x), int(y_bottom)), float(base), 1)
        y = y_top
        while y <= y_bottom:
            r = y - size / 2
            gau = gau_a * math.e**(-r * r / (2 * gau_c * gau_c))
            img_l[int(y) - 1][int(x) - 1] = np.clip(img_l[int(y) - 1][int(x) - 1], (base + (255 - base) * gau), 255)
            y += ratio_y  * 1
        fx += dx
    
    
    img_l = img_l + np.multiply(img_l, img_dt)
    
    img_l = cv2.normalize(img_l, img_l, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
    m = 0.5
    sigma = 0.5
    noise_05 = np.zeros((int(size/5), int(size/5)), np.float32)
    noise_10 = np.zeros((int(size/10), int(size/10)), np.float32)
    noise_25 = np.zeros((int(size/25), int(size/25)), np.float32)
    cv2.randn(noise_05, m, sigma)
    cv2.randn(noise_10, m, sigma)
    cv2.randn(noise_25, m, sigma)
    noise_scaled = cv2.resize(noise_25, (size, size))
    noise = noise_scaled.copy()
    noise_scaled = cv2.resize(noise_10, (size, size))
    noise += noise_scaled
    noise_scaled = cv2.resize(noise_05, (size, size))
    noise += noise_scaled
    img_l = img_l + np.multiply(img_l, noise)
    #cv2.imshow("test", img_l)
    #cv2.waitKey(0)
    img_l = cv2.resize(img_l, (15, 15), 0.0, 0.0, cv2.INTER_LINEAR)
    img_l = cv2.normalize(img_l, img_l, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
    ground = np.zeros((15, 15), np.float32)
    cv2.randn(ground, 0.75, 0.02)
    img_l = ground - img_l
    _,img_mask = cv2.threshold(ground-img_l, 0, 255, cv2.THRESH_BINARY)
    gray = img_mask
    backtorgb = cv2.cvtColor(img_l,cv2.COLOR_BGR2BGRA)
    backtorgb[:, :, 3] = gray
    ID = str(ID)
    cv2.imwrite(clip  +'/daphnia_gallery/' + ID + ".png", 255*backtorgb)


def create_daphnia_image_homography(ID, clip, x, y, seed, angle):
    size = 200
    small_w = 15
    small_h = 15
    epsilon = 0.0001
    t = 0.3
    l = 1.565
    base = 200
    gau_a = 1.0
    gau_c = 12.0
    
    img = np.zeros((size, size), np.uint8)
    
    dx = (2.0 - 2 * epsilon) / (size)
    fx = -1.0 + epsilon
    while fx < 1.0:
        dy = t * ((1.0 + fx)/(1.0 - fx))**((1.0 - l) / (2.0 * (1.0 + l))) * (1.0 - fx * fx)**0.5
        y_top = ratio_y * size * (0.5 - dy)
        y_bottom = ratio_y * size * (0.5 + dy)
        x = ratio_x * (fx / dx + size / 2)
        cv2.line(img, (int(x), int(y_top)), (int(x), int(y_bottom)), float(base), 1)
        fx += ratio_x * dx
    
    img_dt = cv2.distanceTransform(img, distanceType = cv2.DIST_L1, maskSize = 3).astype(np.float32)
    img_l = np.float32(img)
    
    fx = -1.0 + epsilon
    while fx < 1.0:
        dy =  t * ((1.0 + fx)/(1.0 - fx))**((1.0 - l) / (2.0 * (1.0 + l))) * (1.0 - fx * fx)**0.5
        y_top = ratio_y * size * (0.5 - dy)
        y_bottom = ratio_y * size * (0.5 + dy)
        x = ratio_x * (fx / dx + size / 2)
        cv2.line(img_l, (int(x), int(y_top)), (int(x), int(y_bottom)), float(base), 1)
        y = y_top
        while y <= y_bottom:
            r = y - size / 2
            gau = gau_a * math.e**(-r * r / (2 * gau_c * gau_c))
            img_l[int(y) - 1][int(x) - 1] = np.clip(img_l[int(y) - 1][int(x) - 1], (base + (255 - base) * gau), 255)
            y += ratio_y  * 1
        fx += dx
    
    
    img_l = img_l + np.multiply(img_l, img_dt)
    
    img_l = cv2.normalize(img_l, img_l, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
    m = 0.5
    sigma = 0.5
    noise_05 = np.zeros((int(size/5), int(size/5)), np.float32)
    noise_10 = np.zeros((int(size/10), int(size/10)), np.float32)
    noise_25 = np.zeros((int(size/25), int(size/25)), np.float32)
    cv2.randn(noise_05, m, sigma)
    cv2.randn(noise_10, m, sigma)
    cv2.randn(noise_25, m, sigma)
    noise_scaled = cv2.resize(noise_25, (size, size))
    noise = noise_scaled.copy()
    noise_scaled = cv2.resize(noise_10, (size, size))
    noise += noise_scaled
    noise_scaled = cv2.resize(noise_05, (size, size))
    noise += noise_scaled
    img_l = img_l + np.multiply(img_l, noise)
    new_rect = cv2.
    
    
    
    
if __name__ == '__main__':
    create_daphnia_image(0, "test", 0.7, 1, 1337)
    
