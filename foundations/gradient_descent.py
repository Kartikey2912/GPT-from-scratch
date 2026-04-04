class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        fx = init
        for i in range(iterations):
            fxx = 2*fx
            fx = fx - learning_rate * fxx
        return round(fx, 5)
        
