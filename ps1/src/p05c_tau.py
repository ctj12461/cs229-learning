import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    total = len(tau_values)
    fig, ax = plt.subplots(2, 3)
    
    min_mse = np.inf
    min_idx = -1

    for i in range(total):
        tau = tau_values[i]
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        mse = np.linalg.norm(predictions - y_valid, 2) ** 2 / x_valid.shape[0]

        ax[i // 3, i % 3].scatter(x_train[:, 1], y_train, c="b", marker="x", s=3)
        ax[i // 3, i % 3].scatter(x_valid[:, 1], y_valid, c="r", marker="o", s=3)
        ax[i // 3, i % 3].scatter(x_valid[:, 1], predictions, c="k", marker="+", s=3)
        ax[i // 3, i % 3].set_title(f"tau = {tau} MSE = {mse}")

        if mse < min_mse:
            min_mse = mse
            min_idx = i
    
    plt.show()
    
    tau = tau_values[min_idx]
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    predictions = model.predict(x_test)
    mse = np.linalg.norm(predictions - y_test, 2) ** 2 / x_test.shape[0]
    print(f"MSE = {mse}")

    with open(pred_path, "w") as f:
        for pred in predictions:
             f.write(f"{pred}\n")
    # *** END CODE HERE ***
