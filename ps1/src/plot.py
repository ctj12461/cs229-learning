import matplotlib.pyplot as plt
import numpy as np

from p01b_logreg import LogisticRegression
from p01e_gda import GDA
import util

def plot_logistic(x_train, y_train, ax):
    new_x_train = util.add_intercept(x_train)
    model = LogisticRegression()
    model.fit(new_x_train, y_train)
    
    theta = model.theta
    x1 = np.linspace(-1, 7, 200)
    x2 = (-theta[1] / theta[2]) * x1 - theta[0] / theta[2]
    ax.plot(x1, x2, label="Logistic Regression")

def plot_gda(x_train, y_train, ax):
    model = GDA()
    model.fit(x_train, y_train)
    
    sigma_inv = np.linalg.inv(model.sigma)
    mu_0 = model.mu[0]
    mu_1 = model.mu[1]
    a = 2 * (mu_0 - mu_1).transpose() @ sigma_inv
    b = np.inner(mu_0.transpose() @ sigma_inv, mu_0) - np.inner(mu_1.transpose() @ sigma_inv, mu_1)
    x1 = np.linspace(-1, 7, 200)
    x2 = (-a[0] / a[1]) * x1 + b / a[1]
    ax.plot(x1, x2, label="GDA")

def main():
    x_train, y_train = util.load_dataset("../data/ds1_train.csv")
    m = x_train.shape[0]

    positive_x1 = []
    positive_x2 = []
    negative_x1 = []
    negative_x2 = []

    for i in range(m):
        if y_train[i] == 1:
            positive_x1.append(x_train[i][0])
            positive_x2.append(x_train[i][1])
        else:
            negative_x1.append(x_train[i][0])
            negative_x2.append(x_train[i][1])

    fig, ax = plt.subplots()
    ax.scatter(positive_x1, positive_x2, s=3, label="Positive Sample")
    ax.scatter(negative_x1, negative_x2, s=3, label="Negative Sample")
    plot_logistic(x_train, y_train, ax)
    plot_gda(x_train, y_train, ax)

    ax.margins(0.0, 0.0)
    ax.autoscale(enable=None, axis="x", tight=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()