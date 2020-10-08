import cv2
import numpy as np
import glob
 
img_array = []

for filename in glob.glob('60test/*.png'):
	img = cv2.imread(filename)
	print(filename)
	img_array.append(img)
height, width, layers = img_array[1].shape

size = (width,height) 

out = cv2.VideoWriter('60test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width,height))
 
for i in range(len(img_array)):
	out.write(img_array[i])

out.release()