from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sys import argv


def create_linear_regression_and_plot(n: int, _d: int, var: float, results, dontprint=True, exponential=False, lower=-1, upper=1):
    # x is uniform distributed
    xs = 0

    xs = np.random.uniform(lower, upper, n)
    # error normal distributed
    # print(xs)
    errors = np.random.normal(0, var, n)
    ys = 0

    if (exponential):
        ys = np.exp(xs) + errors
    else:
        ys = xs + errors
    # print(ys)

    for d in range(1, _d+1):
        design_matrix = np.ones(n)

        for i in range(1, d+1):
            xpower = np.power(xs, i)
            design_matrix = np.vstack((design_matrix, xpower))

        design_matrix = design_matrix.T

        transformation_matrix = np.linalg.inv(
            design_matrix.T @ design_matrix) @ design_matrix.T @ ys

        y_pred = transformation_matrix[0]
        x_line = np.linspace(lower, upper, 2000)

        for i in range(1, d+1):
            y_pred += transformation_matrix[i] * np.power(x_line, i)

        x_n = np.linspace(0, 1, n)
        y_n = transformation_matrix[0]
        for i in range(1, d+1):
            y_n += transformation_matrix[i] * np.power(x_n, i)

        bias_matrix = y_n - xs
        bias = np.linalg.norm(bias_matrix)
        bias = bias / n

        #print("With degree: ", d, " the bias is: ", bias)
        results[d-1][(n//15) - 1] += bias

        if(not dontprint):
            name = "degree:"+str(d)
            plt.scatter(xs, ys)
            plt.plot(x_line, y_pred, label=name)
            plt.legend()

            figname = "degree-"+str(i)+".png"
            plt.savefig(figname)
            plt.clf()


def main():
    results = np.zeros((15, 20))
    yr = np.zeros(20)
    for i in range(0, 20):
        yr[i] = i*15

    xr = np.zeros(15)
    for i in range(0, 15):
        xr[i] = i+1

    #create_linear_regression_and_plot(20, 10, 0.1, results, False)
    # create_linear_regression_and_plot(
    #   15, 15, 0.1, results, dontprint=False, exponential=True, lower=-3, upper=4)

    for j in range(0, 500):
        print("iteration: ", j, "/500")
        for i in range(1, 21):
            create_linear_regression_and_plot(i*15, 15, 0.1, results)

    np.true_divide(results, 50)

    # plot error of linear
    print(results)

    # take log for better scaling
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xr, yr)
    ha.plot_surface(X.T, Y.T, np.log2(results))
    figname = "lin-lin.png"
    plt.savefig(figname)
    plt.clf()

    # calculate exponential

    results = np.zeros((15, 20))

    for j in range(0, 500):
        print("iteration: ", j, "/500")
        for i in range(1, 21):
            create_linear_regression_and_plot(
                i*15, 15, 0.1, results, exponential=True, lower=-3, upper=4)

    np.true_divide(results, 500)

    # plot error of exponential
    print(results)

    # take log for better scaling
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xr, yr)
    ha.plot_surface(X.T, Y.T, np.log2(results))
    figname = "lin-exp.png"
    plt.savefig(figname)
    plt.clf()


if __name__ == "__main__":
    # execute only if run as a script
    main()
