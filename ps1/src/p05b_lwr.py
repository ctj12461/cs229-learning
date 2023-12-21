import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    predictions = model.predict(x_eval)
    mse = np.linalg.norm(predictions - y_eval, 2) ** 2 / x_eval.shape[0]

    fig, ax = plt.subplots()
    ax.scatter(x_train[:, 1], y_train, c="b", marker="x", s=3)
    ax.scatter(x_eval[:, 1], y_eval, c="r", marker="o", s=3)
    ax.scatter(x_eval[:, 1], predictions, c="k", marker="+", s=3)
    plt.show()

    print(f"MSE = {mse}")
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = self.x.shape[0]
        n = self.x.shape[1]
        res = []

        for xx in x:
            w = np.zeros((m, m))
            
            for i in range(m):
                w[i][i] = np.exp(-(np.linalg.norm(xx - self.x[i], 2) ** 2) / (2 * self.tau ** 2)) / 2

            xw = self.x.transpose() @ w
            theta = (np.linalg.inv(xw @ self.x) @ xw) @ self.y
            res.append(np.inner(theta, xx))

        return res
        # *** END CODE HERE ***
