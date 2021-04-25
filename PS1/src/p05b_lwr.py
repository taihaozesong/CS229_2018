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
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    mse = np.mean((y_pred - y_eval)**2)
    print(f'MSE={mse}')

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        (self.x, self.y) = (x, y)
        # *** END CODE HERE ***

    def getW(self,x,X,tau):
        m = X.shape[0]
        W = np.mat(np.eye(m))
        for i in range(m):
            xi = X[i]
            d = -2*tau*tau
            W[i,i] = np.exp((xi-x).dot(xi-x).T/d)
        return W

    def predict(self, X):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = X.shape
        y_pred = np.zeros(m)
        for i in range(m):
            x = X[i]
            W = self.getW(x, self.x, self.tau)
            p2 = self.x.T.dot(W).dot(np.reshape(self.y, (-1, 1)))
            theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(p2)
            y_pred[i] = theta.T.dot(x)
        return y_pred
        # *** END CODE HERE ***
