import json
import math
import os
import random
import daph_gallery
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import seaborn as sns

from utils import LightSystem

BG_IMG = cv2.imread('samples/bg_sample_1.png', cv2.IMREAD_UNCHANGED)
SEGMENTS = {0: 'center.', 1: 'cells.', 2: 'boundaries.'}
STATUS = {0: 'Non-medicated', 1: 'Medicated'}
GENDER = ['MALE', 'FEMALE']
CLIP_MINS = {0: 75, 1: 0, 2: 45}

LIGHT = {0: 'off.', 1: 'on.'}
LIGHT_ON = 0
# LIGHT_DISTRIBUTION = generate_2d_gauss([100, 100], [[100000, 0], [0, 100000]])
LIGHT_INTENTISY = 0.3
global LIGHT_DISTRIBUTION
# LIGHT_DISTRIBUTION = np.uint8(generate_light_distribution((0, 0), LIGHT_INTENTISY) * 255)
# print(LIGHT_DISTRIBUTION[0])
LIGHT_MIN = 1e-10
LIGHT_MAX = 1
x_light, y_light = np.mgrid[0:1280:1, 0:1024:1]
light_system = LightSystem(LIGHT_INTENTISY)

LIGHT_INTENTISY_SMOOTH = np.arange(0.1, 1.1, 0.1)
global LIGHT_INTENTISY_SMOOTH_INDEXER
LIGHT_INTENTISY_SMOOTH_INDEXER = 1

xFlucts = np.random.uniform(-4, 0, 20)
yFlucts = np.random.uniform(-1, 0, 20)
velFlucts = np.random.uniform(-2, 2, 20)
leftAngles = np.random.normal(2 * math.pi, math.pi / 4, 180)
downAngles = np.random.normal(math.pi / 2, math.pi / 4, 180)
rightAngles = np.random.normal(math.pi, math.pi / 4, 180)
upAngles = np.random.normal(3 * math.pi / 2, math.pi / 4, 180)
bigPlane = np.random.normal(10, 1.25, 30)
smallPlane = np.random.normal(5, 0.3, 30)
launchAngles = np.random.normal(math.pi / 2, math.pi / 6, 180)


leftAngles *= 180 / math.pi
rightAngles *= 180 / math.pi
downAngles *= 180 / math.pi
upAngles *= 180 / math.pi

acceleration = np.random.uniform(0.7, 1, 20)
AFFECTED = np.random.uniform(0.3, 1, 40)
#xGenPos = np.random.uniform(10, 1180, 1000)
xGenPos = np.random.normal(585, 141.25, 1000)
yGenPos = np.random.normal(251, 60.25, 1000)
#yGenPos = np.random.uniform(10, 512, 1000)
heatmap_gen = np.zeros((1024, 1280), dtype = np.float32)
total_heatmap = np.zeros((1024, 1280), dtype = np.float32)

for x in xGenPos:
    for y in yGenPos:
        heatmap_gen[1022 - (int(y) - 1)][int(x + 1)] += 1
        heatmap_gen[1022 - (int(y) - 1)][int(x - 1)] += 1
        heatmap_gen[1022 - (int(y) - 1)][int(x)] += 1
        heatmap_gen[1022 - (int(y - 1) - 1)][int(x + 1)] += 1 
        heatmap_gen[1022 - (int(y - 1) - 1)][int(x - 1)] += 1
        heatmap_gen[1022 - (int(y - 1) - 1)][int(x)] += 1
        heatmap_gen[1022 - (int(y + 1) - 1)][int(x + 1)] += 1
        heatmap_gen[1022 - (int(y + 1) - 1)][int(x - 1)] += 1
        heatmap_gen[1022 - (int(y + 1) - 1)][int(x)] += 1


FRAMES_NO_TURN = 0
FRAMES_NO_TURN_3D = 0

HIGH_STATS = []
LOW_STATS = []
VELOCITY_STATS = []
TURN_STATS = []
POSITION_STATS = []
Ystats = []

MARKED = 0
HIGH_ACCUM = 0
LOW_ACCUM = 0

with open("samples\\larger_sample.json") as f:
    data = json.load(f)
radiusesX = data

with open("samples\\smaller_sample.json") as f:
    data = json.load(f)
radiusesY = data

with open("samples\\velocities.json") as f:
    data = json.load(f)
velocities = data

with open("samples\\turn_rates.json") as f:
    data = json.load(f)
turn_rates = data


def increase_brightness(img, value=None, light_distribution=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value is not None:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    elif light_distribution is not None:
        lim = (np.zeros(light_distribution.shape) + 255) - light_distribution
        v[v > lim] = 255
        v[v <= lim] += light_distribution[v <= lim]

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def switch_LIGHT():
    global LIGHT_ON
    LIGHT_ON += 1
    LIGHT_ON = LIGHT_ON % 2


def frame_TRANSITION(fps_coef):
    global FRAMES_NO_TURN
    global FRAMES_NO_TURN_3D
    FRAMES_NO_TURN = (FRAMES_NO_TURN + 1) % (3 * fps_coef)
    FRAMES_NO_TURN_3D = (FRAMES_NO_TURN_3D + 1) % (3 * fps_coef * 2)


def check_segment(daphnia):
    if daphnia[0][1] >= 964:
        daphnia[3] = 1
    elif daphnia[0][0] <= 50 or daphnia[0][0] >= 1160:
        daphnia[3] = 2
    else:
        daphnia[3] = 0


def gen_daphnia(velocities, fps_coef):
    radiusX = int(random.choice(bigPlane))
    if radiusX > 10:
        radiusX = 10
    #radiusX = 5
    radiusY = random.choice(smallPlane)
    radiusY = int(radiusX * (6/10))
    velocity = random.choice(velocities)
    if velocity > 10 * fps_coef:
        velocity = 10 * fps_coef
    turn_rate = random.randint(0, 360)
    affected = random.choice(AFFECTED)
    center = [int(random.choice(xGenPos)), 1024 - int(random.choice(yGenPos))]
    turned = 0
    if center[1] <= 60:
        segment = 1
    elif center[0] <= 50 or center[0] >= 1160:
        segment = 2
    else:
        segment = 0
    return [center, max(radiusX, radiusY), min(radiusX, radiusY), segment, affected, velocity, turned, turn_rate]


def move_daphnia(daphnia, velocities, turn_rates, fps_coef):
    global HIGH_ACCUM
    global LOW_ACCUM
    v = daphnia[5]
    fluct = random.choice(velFlucts)
    v += fluct
    v *= 1 + random.choice(acceleration) * 0.05 * \
         light_system.current_light_distribution[int(daphnia[0][1]) - 1][int(daphnia[0][0]) - 1] * daphnia[4]
    # v *= 1 + random.choice(acceleration) * LIGHT_ON * LIGHT_BRIGHTNESS*10 * daphnia[4]
    v *= fps_coef
    if MARKED == 1:
        HIGH_ACCUM += v
    if MARKED == 2:
        LOW_ACCUM += v
    if FRAMES_NO_TURN == 0:
        if abs((random.choice(turn_rates) * 180 / math.pi)) < 91:
            daphnia[-1] += (random.choice(turn_rates) * 180 / math.pi) * fps_coef
    
    daphnia[0][0] += v * math.cos(daphnia[-1] * math.pi / 180)
    daphnia[0][1] += v * math.sin(daphnia[-1] * math.pi / 180)
    #if daphnia[0][1] >= 30:
    daphnia[0][1] -= fps_coef * 0.5 * max(0, 5 * fps_coef - v) / (5 * fps_coef)
    if daphnia[0][0] > 1180:
        daphnia[0][0] = 1175
        daphnia[-1] = random.choice(rightAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[0][1] > 990:
        daphnia[0][1] = 985
        daphnia[-1] = random.choice(upAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[0][0] <= 30:
        daphnia[0][0] = 35
        daphnia[-1] = random.choice(leftAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[0][1] <= 15:
        daphnia[0][1] = 20
        if daphnia[6] == 0:
            daphnia[5] = 2 * fps_coef
        daphnia[-1] = random.choice(downAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[3] == 1 and LIGHT_ON == 1 and daphnia[4] == 1 and daphnia[6] == 0:
        if daphnia[5] < 5:
            daphnia[5] = 5 * fps_coef
        daphnia[-1] = random.choice(launchAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef
        daphnia[6] = 1
    check_segment(daphnia)



def gen_daphnias(n, velocities, fps_coef, name):
    daphnias = []
    for i in range(n):
        daphnias.append(gen_daphnia(velocities, fps_coef))
    for i in range(len(daphnias)):
        daph_gallery.create_daphnia_image(i, name, daphnias[i][1]/15, daphnias[i][2]/15, 1337)
    return daphnias

def create_frame(ID, name, prev_frame, velocities, turn_rates, fps_coef):
    yFrame = []
    global VELOCITY_STATS
    global TURN_STATS
    global POSITION_STATS
    global HIGH_ACCUM
    global LOW_ACCUM
    global MARKED 
    HIGH_ACCUM = 0
    LOW_ACCUM = 0
    HIGH_COUNT = 0
    LOW_COUNT = 0
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1280 / 256, 1024 / 256), dpi=256)
    plt.axis('off')
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 1024)
    ax.set_aspect('equal')
    #light_system.increase_brightness(BG_IMG.copy())
    """if LIGHT_ON:
        ax.imshow(increase_brightness(BG_IMG.copy(), light_distribution=LIGHT_DISTRIBUTION))
    else:
        ax.imshow(BG_IMG)"""
    ells = []
    #frame_TRANSITION(1 / fps_coef)
    curr_frame = BG_IMG.copy()
    curr_frame_2 = BG_IMG.copy()
    curr_frame = light_system.increase_brightness(curr_frame)
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    curr_frame_2 = light_system.increase_brightness(curr_frame_2)
    curr_frame_2 = cv2.cvtColor(curr_frame_2, cv2.COLOR_RGB2GRAY)
    for i in range(len(prev_frame)):
        curr = []
        if prev_frame[i][4] < 0.8:
            MARKED = 2
            LOW_COUNT += 1
        else:
            MARKED = 1
            HIGH_COUNT += 1
        total_heatmap[int(prev_frame[i][0][1])-1][int(prev_frame[i][0][0])] += 1
        move_daphnia(prev_frame[i], velocities, turn_rates, fps_coef)
           
        curr_dphn = cv2.imread(name + '/daphnia_gallery/' + str(i) + ".png")
        im_new, mask, w, h, ground = daph_gallery.create_daphnia_image_homography(i, name, [2*prev_frame[i][2], 2*prev_frame[i][1], 90 + prev_frame[i][-1]])
        locs = np.where(mask != False)
        #print(curr_frame[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]])
        all_min = np.amin(curr_frame_2)
        all_max = np.amax(curr_frame_2)
        #mean = cv2.mean(curr_frame_2[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]])
        mean = cv2.mean(curr_frame_2[974:1024, 0:1280])
        minu = np.amin(curr_frame_2[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]])
        var = np.var(curr_frame_2[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]])
        maxu = np.amax(curr_frame_2[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]])
        im_new = np.clip(im_new, all_min, 255)
        #mean = cv2.mean(curr_frame[int(prev_frame[i][0][1]):int(prev_frame[i][0][1]) + h, int(prev_frame[i][0][0]):int(prev_frame[i][0][0]) + w])
        #minu = np.amin(curr_frame[int(prev_frame[i][0][1]):int(prev_frame[i][0][1]) + h, int(prev_frame[i][0][0]):int(prev_frame[i][0][0]) + w])
        #print(mean)
        #print(minu)
        #aaaaa = int(input())
        #im_new = im_new + (255 - mean[0])/3
        #print(mean)
        #im_new = 255 - im_new
        #for i in range(len(im_new)):
        #    for j in range(len(im_new[i])):
        #        im_new[i][j] = np.clip(im_new[i][j], 17, 190())
        #cv2.imshow("wffd", im_new)
        #cv2.waitKey(0)
        #cv2.randn(ground, mean[0], var)
        #print(all_min)
        #print(all_max)
        for k in range(len(ground)):
            for j in range(len(ground[k])):
                #ground[k][j] = mean[0]
                ground[k][j] = mean[0]
        #print(ground)
        #if prev_frame[i][3] == 1:
        #    print(ground)
        #    aaaaa = int(input())
        #cv2.imshow("im_new", im_new)
        #cv2.waitKey(0)        
        for k in range(len(im_new)):
            for j in range(len(im_new[k])):
                im_new[k][j] = np.clip(1.0 / 255 * im_new[k][j] * (ground[k][j] - 0.5 * im_new[k][j]) + 1.0 / 255* (255 - im_new[k][j]) * curr_frame[int(prev_frame[i][0][1]) + k, int(prev_frame[i][0][0]) + j], 0, 255)
                #im_new[k][j] = int(ground[k][j]) - im_new[k][j]
        #if prev_frame[i][3] == 1:

        #cv2.imshow("wffd", im_new
        #cv2.waitKey(0)
        #im_new = im_new - minu
        #cv2.imshow("wffd", im_new)
        #cv2.waitKey(0)        
        #im_new = 255 - im_new
        #mm = np.full((h, w), minu, dtype = np.uint8)
        for k in range(len(im_new)):
            for j in range(len(im_new[k])):
                #sec_value = np.clip(curr_frame[int(prev_frame[i][0][1] + j), int(prev_frame[i][0][0] + k)] - im_new[k][j], 0, 255)
                #im_new[k][j] = min(curr_frame_2[int(prev_frame[i][0][1] + j), int(prev_frame[i][0][0] + k)], np.clip(int(mean[0]) - im_new[k][j], all_min, 255))
                im_new[k][j] = min(curr_frame[int(prev_frame[i][0][1] + j), int(prev_frame[i][0][0] + k)], im_new[k][j])
                #im_new[k][j] = np.clip(im_new[k][j], 75, 195)
                #im_new[k][j] = min(curr_frame[int(prev_frame[i][0][1] + j), int(prev_frame[i][0][0] + k)], sec_value)
        #print(im_new)
        #im_new = np.clip(im_new, CLIP_MINS[prev_frame[i][3]], 150)
        #if prev_frame[i][3] != 0:
        #    im_new = im_new * 1.2
        #    im_new = np.clip(im_new, mean[0]/1.5, mean[0])
        #im_new = 255 - im_new
        #im_new = np.clip(im_new, 0, mean[0])
        #im_new = np.clip(im_new, all_min, all_max)
        #print(im_new)
        #aaaaa = int(input())
        #im_new = np.clip(im_new, 0, 255)
        #cv2.imshow("wffd", im_new)
        #cv2.waitKey(0)        
        #im_new = np.clip(im_new, minu, mean[0])
        #cv2.imshow("wffd", im_new)
        #cv2.waitKey(0)        
        #im_new = 255-im_new
        #cv2.imshow("wffd", im_new)
        #cv2.waitKey(0)        
        #im_new = 255 - im_new
        #if prev_frame[i][3] != 0:
        #    im_new = im_new * 0.75
        #print(locs)
        curr_frame[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]] = im_new[locs[0], locs[1]]
        #curr_frame[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]] = np.clip(curr_frame[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]], 45,  im_new[locs[0], locs[1]])
        #print(im_new[locs[0], locs[1]])
        #print(minu - im_new[locs[0], locs[1]])
        #print(minu) 
        #curr_frame[int(prev_frame[i][0][1]) + locs[1], int(prev_frame[i][0][0]) + locs[0]] = np.amin([minu - im_new[locs[0], locs[1]], minu])
        '''
        if FRAMES_NO_TURN_3D == 0:
            curr.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1] + random.choice(xFlucts * fps_coef),
                                        prev_frame[i][2] + random.choice(yFlucts * fps_coef), prev_frame[i][-1]))
            curr.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1] - 4 + random.choice(xFlucts * fps_coef)*2 ,
                                        prev_frame[i][2]/3.2, prev_frame[i][-1]))
            curr.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1] + random.choice(xFlucts * fps_coef)*2 ,
                                        prev_frame[i][2]/2.5, prev_frame[i][-1]))
            ells.append(curr)
        else:
            curr.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1], prev_frame[i][2], prev_frame[i][-1]))
            curr.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1] - 4, prev_frame[i][2]/3.2, prev_frame[i][-1]))
            curr.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1], prev_frame[i][2]/2.5, prev_frame[i][-1]))
            ells.append(curr)
        '''
        
        VELOCITY_STATS.append(prev_frame[i][5])
        TURN_STATS.append(prev_frame[i][-1] % 360)
        POSITION_STATS.append(SEGMENTS[prev_frame[i][3]])
        yFrame.append(prev_frame[i][0][1])
        '''
    for el in ells:
        ax.add_patch(el[0])
        el[0].set_clip_box(ax.bbox)
        el[0].set_facecolor('black')
        el[0].set_alpha(0.3)
        ax.add_patch(el[2])
        el[2].set_clip_box(ax.bbox)
        el[2].set_facecolor('black')
        el[2].set_alpha(0.4)
        ax.add_patch(el[1])
        el[1].set_clip_box(ax.bbox)
        el[1].set_facecolor('black')
        el[1].set_alpha(0.3)      
        '''
    ID = str(ID)
    while len(ID) < 10:
        ID = '0' + ID
    
    #ax.imshow(light_system.increase_brightness(BG_IMG.copy()))
    #curr_frame = light_system.increase_brightness(cv2.cvtColor(curr_frame, cv2.COLOR_RGBA2RGB))
    #ax.set_title("'" + name + "'@" + str(30 / fps_coef) + " fps, light " + LIGHT[light_system.light_enabled])
    # ax.contourf(x_light, y_light, LIGHT_DISTRIBUTION, alpha = 0.0 + 0.15*LIGHT_ON)
    # ax.pcolormesh(x_light, y_light, LIGHT_DISTRIBUTION, cmap='gray', alpha = 0.0 + 0.15*LIGHT_ON, norm=colors.LogNorm(vmin=LIGHT_MIN, vmax=LIGHT_MAX))
    #plt.gca().set_axis_off()
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                #hspace = 0, wspace = 0)
    #plt.margins(0,0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())    
    #fig.savefig(name + '/frames/' + ID + '.png', dpi=256, bbox_inches='tight', pad_inches = 0)
    plt.close('all')
    cv2.imwrite(name + '/frames/' + ID + '.png', curr_frame)
    Ystats.append(yFrame)
    HIGH_STATS.append(HIGH_ACCUM/(HIGH_COUNT+1))
    LOW_STATS.append(LOW_ACCUM/(LOW_COUNT+1))
    
def create_clip(fps, objects, time, clip_name, velocities, turn_rates, lights_on, lights_off):
    json_file = []
    time_line = np.arange(0, time, 1/fps)
    fps_coef = 20 / fps
    
    if not os.path.exists(clip_name):
        os.makedirs(clip_name)
        
    frames_dir = clip_name + "/frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        
    gallery_dir = clip_name + "/daphnia_gallery"
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir)    
    
    dphns = gen_daphnias(objects, velocities, fps_coef, clip_name)
    j = 0
    
    for i in tqdm.tqdm(range(fps * time), position=0, leave=True):
        curr = []
        ID = str(i)
        while len(ID) < 10:
            ID = '0' + ID        
        frame_TRANSITION(1 / fps_coef)
        if (i == (lights_on[(j % len(lights_on))] * fps)): 
            light_system.light_switch()
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
            for k in range(len(dphns)):
                curr.append({'ID': k, 'center': [dphns[k][0][0], dphns[k][0][1]], 'longer': dphns[k][1], 'shorter': dphns[k][2], 'theta': dphns[k][-1]})
            json_file.append({ID : curr})
        elif (i == (lights_off[(j % len(lights_off))]) * fps):
            light_system.light_switch()
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
            j += 1
            for k in range(len(dphns)):
                curr.append({'ID': k, 'center': [dphns[k][0][0], dphns[k][0][1]], 'longer': dphns[k][1], 'shorter': dphns[k][2], 'theta': dphns[k][-1]})
            json_file.append({ID : curr})
        else:
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
            for k in range(len(dphns)):
                curr.append({'ID': k, 'center': [dphns[k][0][0], dphns[k][0][1]], 'longer': dphns[k][1], 'shorter': dphns[k][2], 'theta': dphns[k][-1]})
            json_file.append({ID : curr})
    
    stats_dir = clip_name + "/stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    line = clip_name + "/gt.json"
    with open(line, 'w') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)    
    
    fig, ax = plt.subplots()
    ax.hist(VELOCITY_STATS, bins=20, density=True)
    ax.set_title("Velocity distribution, in pixels")
    fig.savefig(stats_dir + "/velocity_distribution.png")
    plt.close('all')
    
    fig, ax = plt.subplots()
    ax.hist(TURN_STATS, bins=40, density=True)
    ax.set_title("Orientation distribution, in degrees")
    fig.savefig(stats_dir + "/orientation_distribution.png")
    plt.close('all')
    
    fig, ax = plt.subplots()
    ax.hist(POSITION_STATS, bins=3, density=True)
    ax.set_title("Zones distributions, relative")
    fig.savefig(stats_dir + "/sector_distribution.png")
    plt.close('all')
    
    fig, ax = plt.subplots()
    ax.set_title("y coordinate of daphnias")
    ax.set_ylim(0, 1024)
    ax.set_aspect('auto')
    plots = np.zeros((objects, len(Ystats)), dtype = float)
    for i in range(len(Ystats)):
        for j in range(len(Ystats[i])):
            plots[j][i] = Ystats[i][j]
    for i in range(len(plots)):
        ax.plot(time_line, plots[i], c = 'black', alpha = 0.4, linewidth = 0.5)
    lights = []
    for i in range(len(lights_on)):
        light = patches.Rectangle(((lights_on[i]) , 0), (lights_off[i] - lights_on[i]), 1024, angle = 0.0)
        lights.append(light)
    for light in lights:
        ax.add_patch(light)
        light.set_facecolor('yellow')
        light.set_alpha(0.3)    
    fig.savefig(stats_dir + "/y_coord.png")
    plt.close('all')
    
    fig, ax = plt.subplots()
    ax.set_title("High mobility daphnias")
    ax.set_aspect('auto')
    ax.set_ylim(0, 15)
    ax.plot(time_line, HIGH_STATS, c = "blue")
    lights = []
    for i in range(len(lights_on)):
        light = patches.Rectangle(((lights_on[i]) , 0), (lights_off[i] - lights_on[i]), 15, angle = 0.0)
        lights.append(light)
    for light in lights:
        ax.add_patch(light)
        light.set_facecolor('yellow')
        light.set_alpha(0.3)
    fig.savefig(stats_dir + "/high_mobility.png")
    #plt.show()
    plt.close('all')    
    
    fig, ax = plt.subplots()
    ax.set_title("Low mobility daphnias")
    ax.set_aspect('auto')
    ax.set_ylim(0, 15)
    ax.plot(time_line, LOW_STATS, c = "blue")
    lights = []
    for i in range(len(lights_on)):
        light = patches.Rectangle(((lights_on[i]) , 0), (lights_off[i] - lights_on[i]), 15, angle = 0.0)
        lights.append(light)
    for light in lights:
        ax.add_patch(light)
        light.set_facecolor('yellow')
        light.set_alpha(0.3)        
    fig.savefig(stats_dir + "/low_mobility.png")
    plt.close('all')    
    
    ax = sns.heatmap(heatmap_gen, annot = False, yticklabels=False, xticklabels=False)
    ax.tick_params(left=False, bottom=False)
    ax.figure.savefig(stats_dir + "/spawn_map.png")
    
    ax = sns.heatmap(total_heatmap, annot = False, yticklabels=False, xticklabels=False)
    ax.tick_params(left=False, bottom=False)
    ax.figure.savefig(stats_dir + "/heatmap.png")    
    print("Frames, stats and JSON located at '" + clip_name + "' folder")


if __name__ == '__main__':
    create_clip(20, 30, 10, "dirtest", velocities, turn_rates, [1], [5])
