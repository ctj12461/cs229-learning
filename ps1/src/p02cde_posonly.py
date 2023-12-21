import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Train model using t-labels
    train, train_label = util.load_dataset(train_path, 't', add_intercept=True)
    test, _ = util.load_dataset(test_path, 't', add_intercept=True)

    model1 = LogisticRegression()
    model1.fit(train, train_label)
    predictions = model1.predict(test)

    with open(pred_path_c, "w") as f:
        for pred in np.round(predictions):
            f.write(f"{pred}\n")

    # Train model using y-labels
    train, train_label = util.load_dataset(train_path, 'y', add_intercept=True)
    test, _ = util.load_dataset(test_path, 'y', add_intercept=True)

    model2 = LogisticRegression()
    model2.fit(train, train_label)
    predictions = model2.predict(test)

    with open(pred_path_d, "w") as f:
        for pred in np.round(predictions):
            f.write(f"{pred}\n")

    # Prediction on test dateset with probabilities rescaled
    valid, valid_label = util.load_dataset(valid_path, 'y', add_intercept=True)
    alpha = 0
    count = 0

    for i in range(valid.shape[0]):
        if valid_label[i] == 1:
            count += 1
            alpha += model2.predict(valid[i, :])
    
    alpha /= count
    predictions /= alpha

    with open(pred_path_e, "w") as f:
        for pred in np.round(predictions):
            f.write(f"{min(pred, 1.0)}\n")
    
    # Plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    test, test_label = util.load_dataset(test_path, 't')
    
    for i in range(test.shape[0]):
        for j in range(3):
            if test_label[i] == 1:
                ax[j].scatter(test[i][0], test[i][1], c="k", s=5, label="True")
            else:
                ax[j].scatter(test[i][0], test[i][1], c="b", s=5, label="False")

    def plot_line(ax, k, b, title):
        x = np.linspace(-10, 10, 2)
        y = k * x + b
        ax.plot(x, y, c = "r", label="Decision Boundary")
        ax.set_title(title)

    theta1 = model1.theta
    theta2 = model2.theta
    plot_line(ax[0], -theta1[1] / theta1[2], -theta1[0] / theta1[2], "y-labels")
    plot_line(ax[1], -theta2[1] / theta2[2], -theta2[0] / theta2[2], "t-labels")
    plot_line(ax[2], -theta2[1] / theta2[2], -(theta2[0] + np.log(2 / alpha - 1)) / theta2[2], "t-labels with alpha")
    
    for i in range(3):
        ax[i].set_xlim(-5, 5)
        ax[i].set_ylim(-10, 10)
    
    plt.show()
    # *** END CODER HERE
