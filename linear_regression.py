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
    es = np.random.normal(0, 0.25, 1000)

    # y locations
    distributed = np.add(xs, es)

    plot = plt.scatter(xs, distributed)

    # linear regression on data
    m, b = np.polyfit(xs, distributed, 1)
    plt.plot(xs, m * xs + b, color='green', linewidth=5, markersize=15)

    xm = np.mean(xs)
    ym = np.mean(distributed)
    slope = ym / xm

    plt.plot(xs, slope*xs, color='red', linewidth=5, markersize=15)

    xsum = np.mean(xs)
    ysum = np.mean(distributed)
    slope_sums = ysum / xsum

    plt.plot(xs, slope_sums*xs, color='purple', linewidth=5, markersize=15)

    # opens gui and shows
    # plt.show()
    # I want it in a pdf file though
    plot.figure.savefig('scatterplot-linear.pdf')

    # clear the figure
    plt.clf()


def plot_quadratic():
    # x is uniform distributed
    xs = np.random.uniform(0, 1, 1000)

    # yi= 30(xi−0.25)2(xi−0.75)2+ξi
    es = np.random.normal(0, 0.01, 1000)

    # numpy arrays support element wise sub mul etc
    distributed = 30*(((xs-0.25)*(xs-0.25)) * ((xs - 0.75)*(xs - 0.75))) + es
    quad = plt.scatter(xs, distributed)

    # poly degree 3
    xs2 = np.square(xs)
    xs3 = np.power(xs, 3)
    xs4 = np.square(xs2)
    x_line = np.linspace(0, 1, 100)

    X1 = np.c_[np.ones(1000), xs]
    # * -> element wise mul, @ -> matrix mul
    A1 = np.linalg.inv(X1.T @ X1) @ X1.T @ distributed
    y_pred = A1[0] + A1[1]*x_line
    plt.plot(x_line, y_pred, color="green")

    X2 = np.c_[np.ones(1000), xs, xs2]
    A2 = np.linalg.inv(X2.T @ X2) @ X2.T @ distributed
    y_pred = A2[0] + A2[1]*x_line + A2[2]*np.square(x_line)
    plt.plot(x_line, y_pred, color="blue")

    X3 = np.c_[np.ones(1000), xs, xs2, xs3]
    A3 = np.linalg.inv(X3.T @ X3) @ X3.T @ distributed
    y_pred = A3[0] + A3[1]*x_line + A3[2] * \
        np.square(x_line) + A3[3]*np.power(x_line, 3)
    plt.plot(x_line, y_pred, color="red")

    X4 = np.c_[np.ones(1000), xs, xs2, xs3, xs4]
    A4 = np.linalg.inv(X4.T @ X4) @ X4.T @ distributed
    y_pred = A4[0] + A4[1]*x_line + A4[2] * \
        np.square(x_line) + A4[3]*np.power(x_line, 3) + \
        A4[4]*np.power(x_line, 4)
    plt.plot(x_line, y_pred, color="purple")

    quad.figure.savefig('scatterplot-quadratic.pdf')
    plt.clf()


def main():
    plot_linear()
    plot_quadratic()


if __name__ == "__main__":
    # execute only if run as a script
    main()
