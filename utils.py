import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    print(generate_2d_gauss([0, 0], [[100000, 0], [0, 100000]]))
