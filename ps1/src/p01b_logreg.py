import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    
    with open(pred_path, "w") as f:
        for x in x_eval:
            output = model.predict(x)
            f.write(f"{output}\n")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        import math

        sample_num = x.shape[0]
        feature_num = x.shape[1]
        theta = np.zeros(feature_num)
        epsilon = 1e-5

        # Newton's method
        while True:
            # Calculate the derivative of J(theta)
            derivative_theta = np.zeros(feature_num)

            for j in range(feature_num):
                for i in range(sample_num):
                    h = 1 / (1 + math.exp(-np.matmul(theta, x[i])))
                    derivative_theta[j] += (h - y[i]) * x[i][j]
                
                derivative_theta[j] /= sample_num;
    
            # Calculate the hessian of J(theta)
            hessian_theta = np.zeros((feature_num, feature_num))

            for j in range(feature_num):
                for k in range(j + 1):
                    for i in range(sample_num):
                        h = 1 / (1 + math.exp(-np.matmul(theta, x[i])))
                        hessian_theta[j][k] += x[i][j] * x[i][k] * h * (1 - h)
                    
                    hessian_theta[j][k] /= sample_num
                
                for k in range(j + 1, feature_num):
                    hessian_theta[j][k] = hessian_theta[k][j]
            
            delta_theta = np.matmul(np.linalg.inv(hessian_theta), derivative_theta)
            
            if np.linalg.norm(delta_theta, 1) < epsilon:
                break

            theta -= delta_theta

        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        import math
        h = 1 / (1 + math.exp(-np.matmul(self.theta, x)))
        return float(round(h))
        # *** END CODE HERE ***
