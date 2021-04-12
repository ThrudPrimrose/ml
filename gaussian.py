import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def least_squares(xs, ys, a) -> float:
    it = np.nditer(xs, flags=['f_index'])
    _sum = 0.0
    for x in it:
        _sum += (x * a - ys[it.index]) ^ 2

    return _sum

def plot_linear():
    # x is uniform distributed
    xs = np.random.uniform(0, 1, 1000)
    # error normal distributed
    es = np.random.normal(0.5, 0.25, 1000)

    # y locations
    distributed = np.add(xs, es)

    plot = plt.scatter(xs, distributed)

    # linear regression on data
    m, b = np.polyfit(xs, distributed, 1)
    plt.plot(xs, m * xs + b, color='green', linewidth=5, markersize=15)

    # opens gui and shows
    # plt.show()
    # I want it in a pdf file though
    plot.figure.savefig('scatterplot-linear.pdf')

    #clear the figure
    plt.clf()

def plot_quadratic():
    # x is uniform distributed
    xs = np.random.uniform(0, 1, 1000)

    # yi= 30(xi−0.25)2(xi−0.75)2+ξi
    es = np.random.normal(0, 0.01, 1000)

    # numpy arrays support element wise sub mul etc
    distributed = 30*(((xs-0.25)*(xs-0.25)) * ((xs - 0.75)*(xs - 0.75))) + es
    quad = plt.scatter(xs, distributed)

    quad.figure.savefig('scatterplot-quadratic.pdf')
    plt.clf()

def main():
    plot_linear()
    plot_quadratic()

if __name__ == "__main__":
    # execute only if run as a script
    main()
