import glob

import cv2


def make_clip():
    img_array = []

    for filename in glob.glob('60test/*.png'):
        img = cv2.imread(filename)
        print(filename)
        img_array.append(img)
    height, width, layers = img_array[1].shape

    size = (width, height)

    out = cv2.VideoWriter('60test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


if __name__ == '__main__':
    make_clip()
