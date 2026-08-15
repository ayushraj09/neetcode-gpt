import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b1 = np.array(b1, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        z1 = x@W1.T+b1
        a1 = np.maximum(0, z1)
        y_hat = a1@W2.T+b2
        l = np.mean((y_hat-y_true)**2)

        dl_dyhat = 2*(y_hat-y_true)/len(y_true)
        dl_dw2 = np.outer(dl_dyhat, a1.T)
        dl_db2 = dl_dyhat
        dl_da1 = dl_dyhat @ W2

        da1_dz1 = (z1 > 0).astype(float)
        
        dl_dz1 = dl_da1 * da1_dz1
        dl_dw1 = np.outer(dl_dz1, x.T)
        dl_db1 = dl_dz1

        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return {
            'loss': np.round(l, 4),
            'dW1': np.round(dl_dw1, 4),
            'db1': np.round(dl_db1, 4),
            'dW2': np.round(dl_dw2, 4),
            'db2': np.round(dl_db2, 4)

        }
