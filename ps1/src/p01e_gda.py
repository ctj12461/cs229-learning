import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    x_eval, _ = util.load_dataset(eval_path, add_intercept=False)
    predictions = model.predict(x_eval)

    with open(pred_path, "w") as f:
        for pred in predictions:
             f.write(f"{pred}\n")
    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        sample_num = x.shape[0]
        feature_num = x.shape[1]
        
        # Calculate phi
        positive_sample_num = np.sum(y)
        self.phi = positive_sample_num / sample_num

        # Calculate mu_0 and mu_1
        self.mu = []
        self.mu.append(np.zeros(feature_num))
        self.mu.append(np.zeros(feature_num))
        
        for i in range(sample_num):
                self.mu[int(y[i])] += x[i]
        
        self.mu[0] /= sample_num - positive_sample_num
        self.mu[1] /= positive_sample_num

        # Calculate sigma
        self.sigma = np.zeros((feature_num, feature_num))

        for i in range(sample_num):
            vec = x[i] - self.mu[int(y[i])]
            self.sigma += np.outer(vec, vec)
        
        self.sigma /= sample_num
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        n = x.shape[1]
        sqrt_sigma_det = np.sqrt(np.linalg.det(self.sigma))
        sigma_inv = np.linalg.inv(self.sigma)
        xx_0 = x - self.mu[0]
        xx_1 = x - self.mu[1]
        res = []

        for i in range(m):
            expo_0 = -1 / 2 * np.inner(xx_0[i] @ sigma_inv, xx_0[i])
            prob_0 = np.exp(expo_0) / (np.power(2 * np.pi, n / 2) * sqrt_sigma_det)
            expo_1 = -1 / 2 * np.inner(xx_1[i] @ sigma_inv, xx_1[i])
            prob_1 = np.exp(expo_1) / (np.power(2 * np.pi, n / 2) * sqrt_sigma_det)
            
            if prob_0 > prob_1:
                res.append(0)
            else:
                res.append(1)

        return res
        # *** END CODE HERE
