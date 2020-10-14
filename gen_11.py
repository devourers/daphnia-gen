import json
import math
import os
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils import LightSystem

BG_IMG = cv2.imread('samples/bg_sample.png')
print(BG_IMG.shape)
SEGMENTS = {0: 'center.', 1: 'cells.', 2: 'boundaries.'}
STATUS = {0: 'Non-medicated', 1: 'Medicated'}

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
xGenPos = np.random.uniform(10, 1180, 100)
yGenPos = np.random.uniform(10, 512, 100)

FRAMES_NO_TURN = 0
FRAMES_NO_TURN_3D = 0

VELOCITY_STATS = []
TURN_STATS = []
POSITION_STATS = []

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
    if daphnia[0][1] <= 60:
        daphnia[3] = 1
    elif daphnia[0][0] <= 20 or daphnia[0][0] >= 1260:
        daphnia[3] = 2
    else:
        daphnia[3] = 0


def gen_daphnia(velocities, fps_coef):
    radiusX = random.choice(bigPlane)
    radiusY = random.choice(smallPlane)
    velocity = random.choice(velocities)
    if velocity > 10 * fps_coef:
        velocity = 10 * fps_coef
    turn_rate = random.randint(0, 360)
    affected = random.choice(AFFECTED)
    center = [int(random.choice(xGenPos)), int(random.choice(yGenPos))]
    turned = 0
    if center[1] <= 60:
        segment = 1
    elif center[0] <= 20 or center[0] >= 1260:
        segment = 2
    else:
        segment = 0
    return [center, max(radiusX, radiusY), min(radiusX, radiusY), segment, affected, velocity, turned, turn_rate]


def move_daphnia(daphnia, velocities, turn_rates, fps_coef):
    v = daphnia[5]
    fluct = random.choice(velFlucts)
    v += fluct
    v *= 1 + random.choice(acceleration) * 0.05 * \
         light_system.current_light_distribution[int(daphnia[0][1]) - 1][int(daphnia[0][0]) - 1] * daphnia[4]
    # v *= 1 + random.choice(acceleration) * LIGHT_ON * LIGHT_BRIGHTNESS*10 * daphnia[4]
    v *= fps_coef
    if FRAMES_NO_TURN == 0:
        if abs((random.choice(turn_rates) * 180 / math.pi)) < 91:
            daphnia[-1] += (random.choice(turn_rates) * 180 / math.pi) * fps_coef
    daphnia[0][0] += v * math.cos(daphnia[-1] * math.pi / 180)
    daphnia[0][1] += v * math.sin(daphnia[-1] * math.pi / 180)
    if daphnia[0][1] >= 30:
        daphnia[0][1] -= fps_coef * 0.5 * max(0, 5 * fps_coef - v) / (5 * fps_coef)
    if daphnia[0][0] > 1180:
        daphnia[0][0] = 1175
        daphnia[-1] = random.choice(rightAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[0][1] > 1000:
        daphnia[0][1] = 995
        daphnia[-1] = random.choice(upAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[0][0] <= 20:
        daphnia[0][0] = 25
        daphnia[-1] = random.choice(leftAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[0][1] <= 5:
        daphnia[0][1] = 10
        if daphnia[6] == 0:
            daphnia[5] = 2 * fps_coef
        daphnia[-1] = random.choice(downAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef

    if daphnia[3] == 1 and LIGHT_ON == 1 and daphnia[4] == 1 and daphnia[6] == 0:
        """if daphnia[5] < 10:
            daphnia[5] = 5 * fps_coef"""
        daphnia[-1] = random.choice(launchAngles) + random.choice(turn_rates) * 180 / math.pi * fps_coef
        daphnia[6] = 1
    check_segment(daphnia)


def gen_daphnias(n, velocities, fps_coef):
    daphnias = []
    for i in range(n):
        daphnias.append(gen_daphnia(velocities, fps_coef))
    return daphnias


def create_frame(ID, name, prev_frame, velocities, turn_rates, fps_coef):
    global VELOCITY_STATS
    global TURN_STATS
    global POSITION_STATS
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1280 / 256, 1024 / 256), dpi=256)
    plt.axis('off')
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 1024)
    ax.set_aspect('equal')
    ax.imshow(light_system.increase_brightness(BG_IMG.copy()))
    """if LIGHT_ON:
        ax.imshow(increase_brightness(BG_IMG.copy(), light_distribution=LIGHT_DISTRIBUTION))
    else:
        ax.imshow(BG_IMG)"""
    ells = []
    for i in range(len(prev_frame)):
        move_daphnia(prev_frame[i], velocities, turn_rates, fps_coef)
        frame_TRANSITION(1 / fps_coef)
        if FRAMES_NO_TURN_3D == 0:
            ells.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1] + random.choice(xFlucts * fps_coef),
                                        prev_frame[i][2] + random.choice(yFlucts * fps_coef), prev_frame[i][-1]))
        else:
            ells.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1], prev_frame[i][2], prev_frame[i][-1]))
        VELOCITY_STATS.append(prev_frame[i][5])
        TURN_STATS.append(prev_frame[i][-1] % 360)
        POSITION_STATS.append(SEGMENTS[prev_frame[i][3]])
    for el in ells:
        ax.add_patch(el)
        el.set_clip_box(ax.bbox)
        el.set_facecolor('black')
        el.set_alpha(0.4)
    ID = str(ID)
    while len(ID) < 10:
        ID = '0' + ID
    ax.set_title("'" + name + "'@" + str(30 / fps_coef) + " fps, light " + LIGHT[light_system.light_enabled])
    # ax.contourf(x_light, y_light, LIGHT_DISTRIBUTION, alpha = 0.0 + 0.15*LIGHT_ON)
    # ax.pcolormesh(x_light, y_light, LIGHT_DISTRIBUTION, cmap='gray', alpha = 0.0 + 0.15*LIGHT_ON, norm=colors.LogNorm(vmin=LIGHT_MIN, vmax=LIGHT_MAX))
    fig.savefig(name + '/' + name + '_' + ID + '.png', dpi=256, bbox_inches='tight')
    plt.close('all')


def create_clip(fps, objects, time, clip_name, velocities, turn_rates, light):
    fps_coef = 30 / fps
    print(fps_coef)
    if not os.path.exists(clip_name):
        os.makedirs(clip_name)
    dphns = gen_daphnias(objects, velocities, fps_coef)
    for i in tqdm.tqdm(range(fps * time)):
        # print("Frame " + str(i+1) + "/" + str(fps*time))
        frame_TRANSITION(1 / fps_coef)
        if (i == int(fps * time / 10) or (i == int(8 * fps * time / 10))):
            light_system.light_switch()
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
        else:
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
    stats_dir = clip_name + "_stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].hist(VELOCITY_STATS, bins=20, density=True)
    ax[0].set_title("Velocity distribution, in pixels")
    ax[1].hist(TURN_STATS, bins=40, density=True)
    ax[1].set_title("Orientation distribution, in degrees")
    ax[2].hist(POSITION_STATS, bins=3, density=True)
    ax[2].set_title("Zones distributions, relative")
    fig.savefig(stats_dir + "/" + clip_name + ".png")
    plt.close('all')
    print("done")


if __name__ == '__main__':
    create_clip(60, 30, 10, "60test", velocities, turn_rates, 5)
