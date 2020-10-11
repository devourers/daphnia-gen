from time import time

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


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


def generate_light_distribution(center, intencity, size=(1, 1), w=1280, h=1024):
    print("Light generating starts..")
    start_time = time()
    field = np.zeros((w, h))
    light_source = []
    for i in range(size[0]):
        for j in range(size[1]):
            field[center[0] + i][center[1] + j] = intencity
            light_source.append((i, j))
    for i in range(w):
        for j in range(h):
            for light_point in light_source:
                if (i, j) == light_point:
                    continue
                distance = np.linalg.norm([light_point[0] - i, light_point[1] - j])
                field[i][j] += intencity / (distance * distance)
    print(f"Light generating finished in {time() - start_time}s")
    return field


if __name__ == '__main__':
    # light = generate_2d_gauss([0, 0], [[200000, 0], [0, 200000]])
    intenticty = 1
    light = generate_light_distribution((0, 0), intenticty)
    x, y = np.mgrid[0:1280:1, 0:1024:1]

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    pcm = ax[0].pcolormesh(x, y, light, cmap='gray', norm=colors.LogNorm(vmin=1e-10, vmax=1))
    fig.colorbar(pcm, ax=ax[0], extend='max')

    #light = generate_light_distribution((0, 0), intenticty*1000)
    light *= 1000
    pcm = ax[1].pcolormesh(x, y, light, cmap='gray', norm=colors.LogNorm(vmin=1e-10, vmax=1))
    fig.colorbar(pcm, ax=ax[1], extend='max')
    plt.show()
