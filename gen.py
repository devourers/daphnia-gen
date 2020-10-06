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

def check_segment(daphnia):
    if daphnia[0][1] <= 60:
        daphnia[3] = 1
    elif daphnia[0][0] <= 20 or daphnia[0][0] >= 1260:
        daphnia[3] = 2
    else:
        daphnia[3] = 0    

def gen_daphnia(x, y):
    radiusX = random.choice(x)
    radiusY = random.choice(y)
    velocity = 0
    turn_rate = 0
    affected = random.choice([0, 1])
    center = [random.randint(0, 1280), random.randint(0, 1024)]
    if center[1] <= 60:
        segment = 1
    elif center[0] <= 20 or center[0] >= 1260:
        segment = 2
    else:
        segment = 0
    return [center, radiusX, radiusY, segment, affected, velocity, turn_rate]

def move_daphnia(daphnia, velocities, turn_rates):
    v = random.choice(velocities)
    v *= (1 + LIGHT_ON*daphnia[4])
    daphnia[5] = v
    theta = random.choice(turn_rates)
    daphnia[6] = theta
    daphnia[0][0] += v*math.cos(theta)
    daphnia[0][1] += v*math.sin(theta)
    if daphnia[0][0] > 1280:
        daphnia[0][0] = 1260
    if daphnia[0][1] >1024:
        daphnia[0][1] = 1004
    if daphnia[0][0] < 0:
        daphnia[0][0] = 0
    if daphnia[0][1] < 0:
        daphnia[0][1] = 0    
    check_segment(daphnia)


class lightButton:
    def __init__(self, master):
        global LIGHT_ON
        global LIGHT        
        self.master = master
        self.button = tk.Button(self.master, text="Switch light.", bg='black', fg='green', width=20, command = self.switch_light_button)
        self.status = tk.Label(self.master, bg='black', fg='green', width = 20, text="Light is " + LIGHT[LIGHT_ON])
        self.button.pack()
        self.status.pack()
    def switch_light_button(self):
        global LIGHT_ON
        global LIGHT
        switch_LIGHT()
        self.status.configure(text="Light is " + LIGHT[LIGHT_ON])

class infoBlock:
    def __init__(self, ID, master, daphnia):
        self.ID = ID
        self.daphnia = daphnia
        self.master = master
        self.pos = tk.StringVar()
        self.seg = tk.StringVar()
        self.vel = tk.StringVar()
        self.turn = tk.StringVar()
        self.pos.set("Center: (" + str(self.daphnia[0][0]) + ";" + str(self.daphnia[0][1]) + ")")
        self.seg.set("In " + SEGMENTS[self.daphnia[3]])
        self.vel.set("Velocity: " + str(self.daphnia[5]))
        self.turn.set("Turn rate: " + str(self.daphnia[6]))
        self.blank = tk.Label(self.master, bg='black', fg='green', width=100)
        self.IDbox = tk.Label(self.master, bg='black', fg='green', width=100, text = "ID: " + str(self.ID))
        self.posBox = tk.Label(self.master, bg='black', fg='green', width=100, text = self.pos.get())
        self.segBox = tk.Label(self.master, bg='black', fg='green', width=100, text = self.seg.get())
        self.velBox = tk.Label(self.master, bg='black', fg='green', width=100, text = self.vel.get())
        self.turnBox = tk.Label(self.master, bg='black', fg='green', width=100, text = self.turn.get())
        self.status = tk.Label(self.master, bg='black', fg = 'green', width=100, text = STATUS[self.daphnia[4]])
        #self.blank.pack()
        self.IDbox.pack()
        self.status.pack()
        self.posBox.pack()
        self.segBox.pack()
        self.velBox.pack()
        self.turnBox.pack()
        self.update()
        #self.master.mainloop()
    def update(self):
        #x = threading.Thread(target = move_daphnia, args =(self.daphnia, velocities, turn_rates,))
        #x.start()
        move_daphnia(self.daphnia, velocities, turn_rates)
        self.pos.set("Center: (" + str(self.daphnia[0][0]) + ";" + str(self.daphnia[0][1]) + ")")
        self.seg.set("In " + SEGMENTS[self.daphnia[3]])
        self.vel.set("Velocity: " + str(self.daphnia[5]))
        self.turn.set("Turn rate: " + str(self.daphnia[6]))
        self.posBox.configure(text = self.pos.get())
        self.segBox.configure(text = self.seg.get())
        self.velBox.configure(text = self.vel.get())
        self.turnBox.configure(text = self.turn.get())
        self.master.update_idletasks()
        self.master.after(100, self.update)
        

daphnias = []
for i in range(7):
    daphnias.append(gen_daphnia(radiusesX, radiusesY))

lightButton(root)
tkdaphnias = []
for i in range(len(daphnias)):
    tkdaphnias.append(infoBlock(i, root, daphnias[i]))

root.mainloop()


        
