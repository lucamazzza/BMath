class Numerical:
    """
    A class for numerical methods.

    Methods:
        false_position(func, a, b, tolerance=0.0001, max_iterations=40)
        bisection(func, a, b, tolerance=0.0001, max_iterations=40)
        newton_raphson(func, derivative, initial_guess, tolerance=0.0001, max_iterations=40)
        fixed_point_iteration(func, initial_guess, tolerance=0.0001, max_iterations=40)
        secant(f, x0, x1, tolerance=0.0001, max_iterations=40)
        tangent(f, f1, x0, tolerance=0.0001, max_iterations=40)

    Author: Luca Mazza
    Version: 1.0
    """

    @staticmethod
    def false_position(func, a, b, tolerance=0.0001, max_iterations=40):
        """
        Implements the false position method to find the root of a function.

        Parameters:
            func (function): The function for which the root is to be found.
            a (float): The lower bound for the root.
            b (float): The upper bound for the root.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 40.

        Returns:
            float: The estimated root of the function, if found within the given tolerance.
            None: If no root is found within the maximum number of iterations.
        """
        if func(a) * func(b) >= 0:
            return None
        for _ in range(max_iterations):
            c = (a * func(b) - b * func(a)) / (func(b) - func(a))
            if abs(func(c)) < tolerance:
                return c
            if func(c) * func(a) < 0:
                b = c
            else:
                a = c
        return None

    @staticmethod
    def newton_raphson(func, derivative, initial_guess, tolerance=0.0001, max_iterations=40):
        """
        Implements the Newton-Raphson method to find the root of a function.

        Parameters:
            func (function): The function for which the root is to be found.
            derivative (function): The derivative of the function.
            initial_guess (float): The initial guess for the root.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 40.

        Returns:
            float: The estimated root of the function, if found within the given tolerance.
            None: If no root is found within the maximum number of iterations.
        """
        x = initial_guess
        for _ in range(max_iterations):
            x = x - func(x) / derivative(x)
            if abs(func(x)) < tolerance:
                return x

        return None

    @staticmethod
    def bisection(func, a, b, tolerance=0.0001, max_iterations=40):
        """
        Implements the bisection method to find the root of a function.

        Parameters:
            func (function): The function for which the root is to be found.
            a (float): The lower bound of the interval.
            b (float): The upper bound of the interval.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 40.

        Returns:
            float: The estimated root of the function, if found within the given tolerance.
            None: If no root is found within the maximum number of iterations.
        """
        for _ in range(max_iterations):
            c = (a + b) / 2
            if abs(func(c)) < tolerance:
                return c
            if func(c) * func(a) < 0:
                b = c
            else:
                a = c
        return None

    @staticmethod
    def secant(f, x0, x1, tolerance=0.0001, max_iterations=40):
        """
        Implements the secant method to find the root of a function.

        Parameters:
            f (function): The function for which the root is to be found.
            x0 (float): The first guess for the root.
            x1 (float): The second guess for the root.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 40.

        Returns:
            float: The estimated root of the function, if found within the given tolerance.
            None: If no root is found within the maximum number of iterations.
        """
        for _ in range(max_iterations):
            x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            if abs(f(x2)) < tolerance:
                return x2
            x0 = x1
            x1 = x2
        return None

    @staticmethod
    def tangent(f, f1, x0, tolerance=0.0001, max_iterations=40):
        """
        Implements the tangent method to find the root of a function.

        Parameters:
            f (function): The function for which the root is to be found.
            f1 (function): The derivative of the function.
            x0 (float): The initial guess for the root.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 40.

        Returns:
            float: The estimated root of the function, if found within the given tolerance.
            None: If no root is found within the maximum number of iterations.
        """
        for _ in range(max_iterations):
            x1 = x0 - f(x0)/f1(x0)
            if abs(f(x1)) < tolerance:
                return x1
            x0 = x1
        return None

    @staticmethod
    def fixed_point_iteration(func, initial_guess, tolerance=0.0001, max_iterations=40):
        """
        Implements the fixed point iteration method to find the root of a function.

        Parameters:
            func (function): The function for which the root is to be found.
            initial_guess (float): The initial guess for the root.
            tolerance (float, optional): The tolerance level for convergence. Defaults to 0.0001.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 40.

        Returns:
            float: The estimated root of the function, if found within the given tolerance.
            None: If no root is found within the maximum number of iterations.
        """
        x = initial_guess
        for _ in range(max_iterations):
            x = func(x)
            if abs(x - func(x)) < tolerance:
                return x
        return None
