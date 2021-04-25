import matplotlib.pyplot as plt
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
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model = PoissonRegression(step_size=lr,max_iter=1e4)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    plt.figure()
    plt.plot(y_eval, y_pred, 'rx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')

    model = PoissonRegression()
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def h_theta(self, x):
        return np.exp(x.dot(self.theta))

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)

        count = 0

        while count < self.max_iter:
            theta_old = np.copy(self.theta)
            self.theta += self.step_size * x.T.dot(y-self.h_theta(x)) / m

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                print("Finished poisson fit")
                break
            else:
                count += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self.h_theta(x)
        # *** END CODE HERE ***
