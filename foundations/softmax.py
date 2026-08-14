import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        z_max = max(z)
        # return np.round(your_answer, 4)
        return np.round(np.exp(z-z_max)/np.sum(np.exp(z-z_max)), 4)
