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

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print("the test accuracy is", np.mean((y_pred > 0.5) == y_eval))
    # print(np.mean(y_pred>0.5 == y_eval))
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        count = 0
        m, n = x.shape
        self.theta = np.zeros(n)
        while count < self.max_iter:
            theta_old = np.copy(self.theta)

            h_theta_x = self.h_theta(x)
            H = self.hessian(x, h_theta_x)
            gradient_J_theta = x.T.dot(h_theta_x - y) / m
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
            else:
                count += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.h_theta(x)
        # *** END CODE HERE ***
