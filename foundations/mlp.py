import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        act = x
        for i in range(len(weights)):
            act = np.matmul(act, weights[i]) + biases[i]

            if i < len(weights) - 1:
                act = np.maximum(0, act) #activation function ReLU
            
        return np.round(act, 5)
            
