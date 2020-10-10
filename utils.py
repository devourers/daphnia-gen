from time import time

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from matplotlib import colors


def generate_2d_gauss(center, cov, w=1280, h=1024):
    x, y = np.mgrid[0:w:1, 0:h:1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(center, cov)

    distr = rv.pdf(pos) * 1000000

    """fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, distr)
    plt.show()"""

    return distr


def generate_light_distribution(center, intencity, size=(3,3), w=1280, h=1024):
    print("Light generating starts..")
    start_time = time()
    field = np.zeros((w, h))
    light_source = []
    for i in range(size[0]):
        for j in range(size[1]):
            field[center[0]+i][center[1] + j] = intencity
            light_source.append((i, j))
    for i in range(w):
        for j in range(h):
            for light_point in light_source:
                if (i, j) == light_point:
                    continue
                distance = np.linalg.norm([light_point[0] - i, light_point[1] - j])
                field[i][j] += intencity / (distance*distance)
    print(f"Light generating finished in {time() - start_time}s")
    return field

if __name__ == '__main__':
    #light = generate_2d_gauss([0, 0], [[200000, 0], [0, 200000]])
    light = generate_light_distribution((0, 0), 100)
    x, y = np.mgrid[0:1280:1, 0:1024:1]

    fig1 = plt.figure()
    ax = fig1.add_subplot()
    ax.contourf(x, y, light, cmap='gray', vmax=light.max(), vmin=light.min())
    plt.show()
