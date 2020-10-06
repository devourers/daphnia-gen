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

xFlucts = [0, -1, -2, -3, -4]

FRAMES_NO_TURN = 0
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
    
def frame_TRANSITION():
    global FRAMES_NO_TURN
    FRAMES_NO_TURN = (FRAMES_NO_TURN + 1) % 5

def check_segment(daphnia):
    if daphnia[0][1] <= 60:
        daphnia[3] = 1
    elif daphnia[0][0] <= 20 or daphnia[0][0] >= 1260:
        daphnia[3] = 2
    else:
        daphnia[3] = 0    

def gen_daphnia(x, y, velocities):
    radiusX = 17
    #radiusX = random.choice(x)
    if radiusX > 23:
        radiusX -= 4
    radiusY = 9
    #radiusY = random.choice(y)
    if radiusY > 15:
        radiusY -= 3    
    velocity = random.choice(velocities)
    if velocity > 22:
        velocity -= 3
    turn_rate = random.randint(0, 360)
    affected = random.choice([0, 1])  
    center = [random.randint(0, 1280), random.randint(0, 512)]
    if center[1] <= 60:
        segment = 1
    elif center[0] <= 20 or center[0] >= 1260:
        segment = 2
    else:
        segment = 0
    return [center, max(radiusX, radiusY), min(radiusX, radiusY), segment, affected, velocity, turn_rate]

def move_daphnia(daphnia, velocities, turn_rates, fps_coef):
    #v = random.choice(velocities)
    v = daphnia[5]
    fluct = random.choice([-2, -1, 0, 1, 2,])
    v += fluct
    v *= (1 + LIGHT_ON*daphnia[4])
    v *= fps_coef
    #daphnia[5] = v
    frame_TRANSITION()
    if FRAMES_NO_TURN == 0:
        if abs((random.choice(turn_rates)*180/math.pi)) < 60:
            daphnia[-1] += (random.choice(turn_rates)*180/math.pi)/fps_coef
    daphnia[0][0] += v*math.cos(daphnia[-1]*math.pi/180)
    daphnia[0][1] += v*math.sin(daphnia[-1]*math.pi/180)
    if daphnia[0][0] > 1280:
        daphnia[0][0] = 1260
        velocity = 3
    if daphnia[0][1] >1024:
        daphnia[0][1] = 1004
        velocity = 3
    if daphnia[0][0] < 0:
        daphnia[0][0] = 0
        velocity = 3
    if daphnia[0][1] < 0:
        daphnia[0][1] = 0
        velocity = 3
    check_segment(daphnia)
    
def gen_daphnias(n, radiusesX, radiusesY, velocities):
    daphnias = []
    for i in range(n):
        daphnias.append(gen_daphnia(radiusesX, radiusesY, velocities))
    return daphnias
    
def create_frame(ID, name, prev_frame, velocities, turn_rates, fps_coef):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 1024)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.figure(figsize = (1280/256, 1024/256), dpi = 256)
    ells = []
    for i in range(len(prev_frame)):
        move_daphnia(prev_frame[i], velocities, turn_rates, fps_coef)
        ells.append(patches.Ellipse(prev_frame[i][0], prev_frame[i][1]+random.choice(xFlucts), prev_frame[i][2], prev_frame[i][-1]))
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
    import os
    if not os.path.exists(clip_name):
        os.makedirs(clip_name)
    dphns = gen_daphnias(objects, radiusesX, radiusesY, velocities)
    for i in range(fps*time):
        if i >= fps*(light-1):
            switch_LIGHT()
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
            switch_LIGHT()
        else:
            create_frame(i, clip_name, dphns, velocities, turn_rates, fps_coef)
    print("done")



create_clip(30, 30, 10, "zlp", velocities, turn_rates, radiusesX, radiusesY, 5)
