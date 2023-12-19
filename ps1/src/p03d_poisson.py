import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = PoissonRegression()
    model.lr = lr
    model.fit(x_train, y_train)

    x_eval, _ = util.load_dataset(eval_path, add_intercept=False)
    predictions = model.predict(x_eval)

    with open(pred_path, "w") as f:
        for pred in predictions:
             f.write(f"{pred}\n")
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        n = x.shape[1]
        epsilon = 1e-5
        theta = np.zeros(n)

        while True:
            change = self.lr * (x.transpose() @ (y - np.exp(x @ theta))) / m

            if np.linalg.norm(change, 1) < epsilon:
                break
            
            theta += change
        
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(self.theta @ x.transpose())
        # *** END CODE HERE ***
