from time import time

import cv2
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def generate_2d_gauss(center, cov, w=1024, h=1280):
    x, y = np.mgrid[0:w:1, 0:h:1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(center, cov)

    distr = rv.pdf(pos)

    """fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, distr)
    plt.show()"""

    return distr


def generate_light_distribution(center, intencity, size=(1, 1), w=1280, h=1024):
    print("No .ld file found, generating starts")
    start_time = time()
    field = np.zeros((h, w))
    light_source = []
    max_dist = np.linalg.norm([h, h])
    for i in range(size[0]):
        for j in range(size[1]):
            field[center[0] + i][center[1] + j] = intencity
            light_source.append((center[0] + i, center[1] + j))
    for i in range(h):
        for j in range(w):
            for light_point in light_source:
                if (i, j) == light_point:
                    continue
                distance = np.linalg.norm([light_point[0] - i, light_point[1] - j])
                # field[i][j] += intencity / (distance*distance)
                if max_dist - distance <= 0:
                    continue
                field[i][j] += intencity * ((max_dist - distance) / max_dist)
    print(f"Light generating finished in {time() - start_time}s")
    with open("samples/light.ld", 'wb') as f:
        np.save(f, field)
    return field
    

class LightSystem:
    def __init__(self, intensity, center=(0, 0)):
        try:
            with open("samples/light.ld", 'rb') as f:
                self.light_distribution = np.load(f)
            print("Light loaded.")
        except:
            self.light_distribution = generate_light_distribution(center, intensity)
        self.light_enabled = 0
        self.light_indexer = 0
        self.current_light_distribution = None
        self.light_enable_sequence = np.arange(0, 1.1, 0.1)
        self.current_light_modifier = 'none'

    def light_switch(self):
        self.light_enabled += 1
        self.light_enabled = self.light_enabled % 2
        if self.light_enabled:
            self.current_light_modifier = 'up'
        else:
            self.current_light_modifier = 'down'

    def light_modify_(self):
        if self.current_light_modifier == 'up':
            self.light_indexer += 1
            #self.current_light_distribution = self.light_distribution * self.light_enable_sequence[self.light_indexer]
            if self.light_indexer == 10:
                self.current_light_modifier = 'none'

        elif self.current_light_modifier == 'down':
            self.light_indexer -= 1
            #self.current_light_distribution = self.light_distribution * self.light_enable_sequence[
            #   self.light_indexer]
            if self.light_indexer == 0:
                self.current_light_modifier = ' none'

        self.current_light_distribution = np.random.uniform(low=0.7, high=1) * self.light_distribution * \
                                          self.light_enable_sequence[self.light_indexer]

        self.current_light_distribution = self.current_light_distribution * 255

    def increase_brightness(self, img, value=None, exposition=True):
        self.light_modify_()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value is not None:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        elif self.current_light_distribution is not None:
            if exposition:
                max_brightness = np.zeros(self.current_light_distribution.shape) + 255
                exposition_coeffs = (1*np.ones(max_brightness.shape) - v/max_brightness)
                lim = max_brightness - exposition_coeffs * self.current_light_distribution
                v[v > lim] = 255
                v[v <= lim] += np.uint8(exposition_coeffs[v <= lim] * self.current_light_distribution[v <= lim])
            else:
                max_brightness = np.zeros(self.current_light_distribution.shape) + 255
                lim = max_brightness - self.current_light_distribution
                v[v > lim] = 255
                v[v <= lim] += np.uint8(self.current_light_distribution[v <= lim])

        hsv = cv2.merge((h, s, v))
        #return cv2.flip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    # light = generate_2d_gauss([0, 0], [[200000, 0], [0, 200000]])
    intenticty = 1
    light = generate_light_distribution((0, 0), intenticty)
    x, y = np.mgrid[0:1280:1, 0:1024:1]

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    pcm = ax[0].pcolormesh(x, y, light, cmap='gray', norm=colors.LogNorm(vmin=1e-10, vmax=1))
    fig.colorbar(pcm, ax=ax[0], extend='max')

    # light = generate_light_distribution((0, 0), intenticty*1000)
    light *= 1000
    pcm = ax[1].pcolormesh(x, y, light, cmap='gray', norm=colors.LogNorm(vmin=1e-10, vmax=1))
    fig.colorbar(pcm, ax=ax[1], extend='max')
    plt.show()
