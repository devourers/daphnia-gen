import glob
import tqdm
import cv2


def make_clip(clip):
    line = clip + '/frames/*.png'
    img_array = []
    for filename in tqdm.tqdm(glob.glob(line), position=0, leave=True):
        img = cv2.imread(filename)
        #print(filename)
        img_array.append(img)
    height, width, layers = img_array[1].shape

    size = (width, height)
    line = clip + '/clip.avi'
    out = cv2.VideoWriter(line, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    print("Clip generated at '" + clip + "' folder.")


if __name__ == '__main__':
    make_clip('dirtest')
