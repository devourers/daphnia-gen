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
    y_top_saved = 100
    y_bot_saved = 100
    
    img = np.zeros((size, size), np.uint8)
    
    dx = (2.0 - 2 * epsilon) / (size)
    fx = -1.0 + epsilon
    while fx < 1.0:
        dy = t * ((1.0 + fx)/(1.0 - fx))**((1.0 - l) / (2.0 * (1.0 + l))) * (1.0 - fx * fx)**0.5
        y_top = size * (0.5 - dy)
        if y_top < y_top_saved:
            y_top_saved = y_top
        y_bottom = size * (0.5 + dy)
        if y_bottom > y_bot_saved:
            y_bot_saved = y_bottom
        x = (fx / dx + size / 2)
        cv2.line(img, (int(x), int(y_top)), (int(x), int(y_bottom)), float(base), 1)
        fx += ratio_x * dx
    
    img_dt = cv2.distanceTransform(img, distanceType = cv2.DIST_L1, maskSize = 3).astype(np.float32)
    img_l = np.float32(img)
    
    fx = -1.0 + epsilon
    while fx < 1.0:
        dy =  t * ((1.0 + fx)/(1.0 - fx))**((1.0 - l) / (2.0 * (1.0 + l))) * (1.0 - fx * fx)**0.5
        y_top = size * (0.5 - dy)
        y_bottom = size * (0.5 + dy)
        x = (fx / dx + size / 2)
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
    img_l = cv2.normalize(img_l, img_l, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
    #cv2.imshow("test", img_l)
    #cv2.waitKey(0)
    #img_l = cv2.resize(img_l, (15, 15), 0.0, 0.0, cv2.INTER_LINEAR)
    
    #ground = np.zeros((15, 15), np.float32)
    #cv2.randn(ground, 0.75, 0.02)
    #img_l = ground - img_l
    #_,img_mask = cv2.threshold(ground-img_l, 0, 255, cv2.THRESH_BINARY)
    #gray = img_mask
    #backtorgb = cv2.cvtColor(img_l,cv2.COLOR_BGR2BGRA)
    #backtorgb[:, :, 3] = gray
    ID = str(ID)
    #cv2.imwrite(clip  +'/daphnia_gallery/' + ID + ".png", 255*img_l)
    cv2.imwrite("test_orig.png", 255*img_l)
    outfile = [y_top_saved, y_bot_saved]
    #with open(clip + '/daphnia_gallery/' + ID + ".dphn", 'wb') as f:
    with open("ys.dphn", 'wb') as f:
        np.save(f, outfile)
    

def find_points(a, b, angle):
    points = []
    points.append([int(25 + 0.5 * b * math.sin(angle * math.pi / 180) + 0.5 * a * math.sin(0.5 * math.pi + math.pi * angle / 180)), int(25 - 0.5 * b * math.cos(math.pi * angle / 180) + 0.5 * a * math.cos(0.5 * math.pi + math.pi * angle / 180))])
    points.append([int(points[0][0] - b * math.sin(angle * math.pi / 180)), int(points[0][1] + b * math.sin(angle * math.pi / 180))])
    points.append([int(points[1][0] - a * math.sin(0.5 * math.pi + angle * math.pi / 180)), int(points[1][1] - a * math.sin(0.5 * math.pi + angle * math.pi / 180))])
    points.append([int(points[0][0] - a * math.sin(0.5 * math.pi + angle * math.pi / 180)), int(points[0][1] - a * math.sin(0.5 * math.pi + angle * math.pi / 180))])
    return [[points[0][1], points[0][0]], [points[1][1], points[1][0]], [points[2][1], points[2][0]], [points[3][1], points[3][0]]]

def create_daphnia_image_homography(ID, clip, measures):
    ID = str(ID)
    #orig = cv2.imread(clip  +'/daphnia_gallery/' + ID + ".png")
    orig = cv2.imread("test_orig.png")
    #with open(clip + '/daphnia_gallery/' + ID + ".dphn", 'rb') as f:
    with open("ys.dphn", 'rb') as f:
        saved = np.load(f)
    dest = np.zeros((50, 50), np.float32)
    if measures[2] > 90:
        measures[2] = measures[2] % 90
    dest_points = find_points(measures[0], measures[1], measures[2])
    print(dest_points)
    orig_points = [[0, saved[1]], [200, saved[1]], [200, saved[0]], [0, saved[0]]]
    dp = cv2.UMat(np.array(dest_points, dtype=np.uint8))
    op = cv2.UMat(np.array(orig_points, dtype=np.uint8))
    h, status = cv2.findHomography(op, dp)
    im_out = cv2.warpPerspective(orig, h, (dest.shape[1],dest.shape[0]))
    im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2GRAY)
    cv2.imshow("test", im_out)
    cv2.waitKey(0)
    ret,thresh = cv2.threshold(im_out,1,255,0)
    cv2.imshow("bbox", ret)
    cv2.waitKey(0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    im_new = im_out[y:y+h, x:x+w]
    mask = im_new > 0
    #print(mask)
    #mask = cv2.UMat(np.array((mask), np.uint8))
    #m = np.asarray(mask)
    #print(m)
    #cv2.imshow("fadcx", mask)
    ground = np.zeros((h, w), np.uint8)
    cv2.randn(ground, 5, 0.2)
    im_new = ground - im_new
    #cv2.imshow("dbfvd", im_new)
    #cv2.waitKey(0)    
    return [im_new, mask, w, h]
    
    
    
    
    
if __name__ == '__main__':
    create_daphnia_image(0, "test", 1, 1, 1337)
    create_daphnia_image_homography(0, "test", [10, 20, 67])
    
