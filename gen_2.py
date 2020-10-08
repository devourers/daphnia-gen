import os
import math
import json
import random
import tkinter as tk
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from scipy.spatial.distance import directed_hausdorff
import threading

root = tk.Tk()

SEGMENTS = {0 : 'center.', 1 : 'cells.', 2 : 'boundaries.'}
STATUS = {0: 'Non-medicated', 1: 'Medicated'}

LIGHT = {0: 'off.', 1: 'on.'}
LIGHT_ON = 0

#xFlucts = [0, -6, -7, -8]
xFlucts = np.random.uniform(-8, 0, 20)
velFlucts = np.random.uniform(-2, 2, 20)
leftAngles = np.random.normal(2*math.pi, math.pi/4, 180)
downAngles = np.random.normal(math.pi/2, math.pi/4, 180)
rightAngles = np.random.normal(math.pi, math.pi/4, 180)
upAngles = np.random.normal(3*math.pi/2, math.pi/4, 180)

leftAngles *= 180 / math.pi
rightAngles *= 180 / math.pi
downAngles *= 180 / math.pi
upAngles *= 180 / math.pi

acceleration = np.random.uniform(0.7, 1, 20)

FRAMES_NO_TURN = 0
FRAMES_NO_TURN_3D = 0

VELOCITY_STATS =[]
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


def switch_LIGHT():
    global LIGHT_ON
    LIGHT_ON += 1
    LIGHT_ON = LIGHT_ON % 2
    
def frame_TRANSITION(fps_coef):
    global FRAMES_NO_TURN
    global FRAMES_NO_TURN_3D
    FRAMES_NO_TURN = (FRAMES_NO_TURN + 1) % (3*fps_coef)
    FRAMES_NO_TURN_3D = (FRAMES_NO_TURN_3D + 1) % (3*fps_coef*2)

def check_segment(daphnia):
    if daphnia[0][1] <= 60:
        daphnia[3] = 1
    elif daphnia[0][0] <= 20 or daphnia[0][0] >= 1260:
        daphnia[3] = 2
    else:
        daphnia[3] = 0    

def gen_daphnia(x, y, velocities, fps_coef):
    radiusX = 17
    #radiusX = random.choice(x)
    if radiusX > 23:
        radiusX -= 4
    radiusY = 9
    #radiusY = random.choice(y)
    if radiusY > 15:
        radiusY -= 3    
    velocity = random.choice(velocities)
    if velocity > 10*fps_coef:
        velocity = 10*fps_coef
    turn_rate = random.randint(0, 360)
    affected = random.choice([0, 1])  
    center = [random.randint(0, 1280), random.randint(0, 512)]
    turned = 0
    if center[1] <= 60:
        segment = 1
    elif center[0] <= 20 or center[0] >= 1260:
        segment = 2
    else:
        segment = 0
    return [center, max(radiusX, radiusY), min(radiusX, radiusY), segment, affected, velocity, turned, turn_rate]

def move_daphnia(daphnia, velocities, turn_rates, fps_coef):
    #v = random.choice(velocities)
    v = daphnia[5]
    fluct = random.choice(velFlucts*fps_coef)
    v += fluct
    v *= (1 + random.choice(acceleration) * LIGHT_ON * daphnia[4])
    v *= fps_coef
    #daphnia[5] = v
    #frame_TRANSITION()
    if FRAMES_NO_TURN == 0:
        if abs((random.choice(turn_rates) * 180 / math.pi)) < 91:
            daphnia[-1] += (random.choice(turn_rates) * 180 / math.pi) * fps_coef
    daphnia[0][0] += v * math.cos(daphnia[-1] * math.pi / 180)
    daphnia[0][1] += v * math.sin(daphnia[-1] * math.pi / 180)
    if daphnia[0][1] >= 30:
        daphnia[0][1] -= fps_coef * 0.5 * max(0, 5 * fps_coef - v)/(5*fps_coef)
    if daphnia[0][0] > 1280:
        daphnia[0][0] = 1275
        #daphnia[5] = 1
        daphnia[-1] = random.choice(rightAngles) + random.choice(turn_rates)*180/math.pi * fps_coef
    if daphnia[0][1] > 1024:
        daphnia[0][1] = 1019
        #daphnia[5]  = 1
        daphnia[-1] = random.choice(upAngles) + random.choice(turn_rates)*180/math.pi * fps_coef
    if daphnia[0][0] <= 0:
        daphnia[0][0] = 5
        #daphnia[5] = 1
        daphnia[-1] = random.choice(leftAngles) + random.choice(turn_rates)*180/math.pi * fps_coef
    if daphnia[0][1] <= 0:
        daphnia[0][1] = 5
        if daphnia[6] == 0:
            daphnia[5] = 2 * fps_coef
        daphnia[-1] = random.choice(downAngles) + random.choice(turn_rates)*180/math.pi * fps_coef
    if daphnia[3] == 1  and LIGHT_ON == 1 and daphnia[4] == 1:
        if daphnia[5] < 10:
            daphnia[5] = 5 * fps_coef
        daphnia[-1] = random.choice(downAngles) + random.choice(turn_rates)*180/math.pi * fps_coef
        daphnia[6] = 1
    check_segment(daphnia)
    
def gen_daphnias(n, radiusesX, radiusesY, velocities, fps_coef):
    daphnias = []
    for i in range(n):
        daphnias.append(gen_daphnia(radiusesX, radiusesY, velocities, fps_coef))
    return daphnias
    
def create_frame(ID, name, prev_frame, velocities, turn_rates, fps_coef):
    global VELOCITY_STATS
    global TURN_STATS
    global POSITION_STATS
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 1024)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.figure(figsize = (1280/256, 1024/256), dpi = 256)
    ells = []
    for i in range(len(prev_frame)):
        move_daphnia(prev_frame[i], velocities, turn_rates, fps_coef)
        frame_TRANSITION(1/fps_coef)
        if FRAMES_NO_TURN_3D== 0:
            ells.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1]+random.choice(xFlucts*fps_coef), prev_frame[i][2], prev_frame[i][-1]))
        else:
            ells.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1], prev_frame[i][2], prev_frame[i][-1]))
        VELOCITY_STATS.append(prev_frame[i][5])
        TURN_STATS.append(prev_frame[i][-1] % 360)
        POSITION_STATS.append(SEGMENTS[prev_frame[i][3]])
    for el in ells:
        ax.add_patch(el)
        el.set_clip_box(ax.bbox)
        el.set_facecolor('black')
    ID = str(ID)
    while len(ID) < 10:
        ID = '0' + ID
    fig.savefig(name + '/' + name + '_' + ID + '.png')
    plt.close()


def create_clip(fps, objects, time, clip_name, velocities, turn_rates, radiusesX, radiusesY, light):
    fps_coef = 30/fps
    print(fps_coef)
    if not os.path.exists(clip_name):
        os.makedirs(clip_name)
    dphns = gen_daphnias(objects, radiusesX, radiusesY, velocities, fps_coef)
    for i in range(fps*time):
        frame_TRANSITION(1/fps_coef)
        if i >= fps*(light-1):
            switch_LIGHT()
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
            switch_LIGHT()
        else:
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
    stats_dir = clip_name + "_stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].hist(VELOCITY_STATS, bins = 20, density = True)
    ax[0].set_title("Velocity distribution, in pixels")
    ax[1].hist(TURN_STATS, bins = 40, density = True)
    ax[1].set_title("Orientation distribution, in degrees")
    ax[2].hist(POSITION_STATS, bins = 3, density = True)
    ax[2].set_title("Zones distributions, relative")
    fig.savefig(stats_dir + "/" + clip_name + ".png")
    plt.close()
    print("done")



create_clip(60, 30, 10, "60test", velocities, turn_rates, radiusesX, radiusesY, 5)
