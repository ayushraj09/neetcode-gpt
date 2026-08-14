import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        
        sum_log=0
        for i in range(len(y_true)):
            if(y_true[i] == 1):
                sum_log += (np.log(y_pred[i])) + 1e-7
            else:
                sum_log += (np.log(1 - y_pred[i])) + 1e-7

        # return round(your_answer, 4)
        return np.round(-1*(np.sum(sum_log))/len(y_true), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        m = len(y_true[0])

        sum_log = 0
        for i in range(n):
            for j in range(m):
                if(y_true[i][j] == 1):
                    sum_log += np.log(y_pred[i][j]) + 1e-7

        return np.round(-1*(np.sum(sum_log))/n, 4)


        
