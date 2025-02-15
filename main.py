import re
import numpy as np
import matplotlib.pyplot as plt
import sympy

from typing import List, Tuple, Union, Callable


class UserInterface:
    def __init__(self) -> None:
        self.function = None
        self.interval = None
        self.degree = None
        self.file_path = None
        self.approaches = []
        
        self.evaluation_points = []

    def get_function_input(self) -> None:
        self.function = input(
            "Enter a function of a single variable (e.g., x*sin(x) - x**2 + 1): ")

    def get_interval_input(self) -> None:
        while True:
            try:
                interval_str = input(
                    "Enter the interval in the format [a, b]: ")
                match = re.match(
                    r'\[([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\]', interval_str)
                if match:
                    self.interval = (float(match.group(1)),
                                     float(match.group(2)))
                    break
                else:
                    print("Invalid format. Please use [a, b].")
            except ValueError:
                print("Invalid input. Please enter numerical values.")

    def get_degree_input(self) -> None:
        while True:
            try:
                self.degree = int(input("Enter the polynomial degree: "))
                if self.degree >= 0:
                    break
                else:
                    print("Degree must be a non-negative integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def get_file_input(self) -> None:
        self.file_path = input("Enter the file path containing 2D points: ")

    def get_approaches_input(self) -> None:
        approaches_dict = {
            "1": "Interpolation SLE",
            "2": "Lagrange Formula",
            "3": "Parametric Interpolation"
        }
        print("Select one or more approaches (comma-separated):")
        for key, value in approaches_dict.items():
            print(f"{key}: {value}")
        choices = input("Enter choices (e.g., 1,2): ").split(',')
        self.approaches = [approaches_dict[choice.strip()]
                           for choice in choices if choice.strip() in approaches_dict]

    def get_evaluation_points(self) -> None:
        while True:
            try:
                points_str = input(
                    "Enter points to evaluate (comma-separated, or 'done' to finish): ")
                if points_str.lower() == 'done':
                    break
                points = [float(x.strip()) for x in points_str.split(',')]

                # Validate points are within interval if function mode
                if self.interval:
                    a, b = self.interval
                    valid_points = [p for p in points if a <= p <= b]
                    if len(valid_points) != len(points):
                        print(
                            f"Some points were outside the interval [{a}, {b}] and were ignored.")
                    points = valid_points

                self.evaluation_points.extend(points)
            except ValueError:
                print(
                    "Invalid input. Please enter numerical values separated by commas.")

    def get_user_input(self) -> None:
        mode = input("Choose input method: (1) Function, (2) File: ")
        if mode == "1":
            self.get_function_input()
            self.get_interval_input()
            self.get_degree_input()
        elif mode == "2":
            self.get_file_input()
            self.get_degree_input()
        else:
            print("Invalid selection.")
            return

        self.get_approaches_input()
        evaluate = input(
            "Would you like to evaluate polynomials at specific points? (y/n): ")
        if evaluate.lower() == 'y':
            self.get_evaluation_points()

    def display_user_input(self) -> None:
        print("\nUser Input:")
        if self.function:
            print(f"Function: {self.function}")
            print(f"Interval: {self.interval}")
        if self.file_path:
            print(f"File Path: {self.file_path}")
        print(f"Polynomial Degree: {self.degree}")
        print(f"Selected Approaches: {', '.join(self.approaches)}")
        if self.evaluation_points:
            print(f"Evaluation Points: {self.evaluation_points}")

    def return_user_input(self) -> Tuple[str, Tuple[float, float], int, Union[str, None], List[str], List[float]]:
        return self.function, self.interval, self.degree, self.file_path, self.approaches, self.evaluation_points


class Interpolation:
    def __init__(self,
                 function: str,
                 interval: Tuple[float, float],
                 degree: int,
                 file_path: Union[str, None],
                 approaches: List[str]) -> None:
        self.X = sympy.Symbol('x')
        self.function = sympy.simplify(sympy.simplify(function))
        self.interval = interval
        self.degree = degree
        self.file_path = file_path
        self.approaches = approaches

        self.interpolation_functions = {}

    def generator_points(self) -> Tuple[List[float], List[float]]:
        num_points = self.degree - 1
        a, b = self.interval
        x_points = np.linspace(a, b, num_points)
        y_points = np.array(
            [float(self.function.subs(self.X, x_subs)) for x_subs in x_points])

        return x_points, y_points

    def interpolation_SLE(self, x_points, y_points) -> Callable[[float], float]:
        if self.file_path:
            data = np.genfromtxt(fname=self.file_path, delimiter=',')
            x_points = data[:, 0]
            y_points = data[:, 1]

        X_Matrix = []
        for i in range(len(x_points)):
            res = []
            for j in range(len(x_points)):
                res.append(x_points[i] ** j)
            X_Matrix.append(res.copy())

        coeffician = np.linalg.solve(X_Matrix, y_points)

        def P(x: float) -> float:
            res = 0
            for i in range(len(coeffician)):
                res += coeffician[i] * x ** i

            return res

        return P

    def Lagrange(self, x_points, y_points) -> Callable[[float], float]:
        if self.file_path:
            data = np.genfromtxt(fname=self.file_path, delimiter=',')
            x_points = data[:, 0]
            y_points = data[:, 1]

        def L_basis(index: int, x: float) -> float:
            numerator = 1
            denominator = 1

            for i in range(len(x_points)):
                if i == index:
                    continue

                numerator *= x - x_points[i]
            for i in range(len(x_points)):
                if i == index:
                    continue

                denominator *= x_points[index] - x_points[i]

            return numerator/denominator

        def P(x: float) -> float:
            res = 0
            for i in range(len(y_points)):
                res += y_points[i] * L_basis(i, x)
            return res

        return P

    def parametric_interpolation(self, x_points, y_points) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        if self.file_path:
            data = np.genfromtxt(fname=self.file_path, delimiter=',')
            x_points = data[:, 0]
            y_points = data[:, 1]

        a, b = self.interval
        t_values = np.linspace(a, b, len(x_points))

        x_t = self.Lagrange(t_values, x_points)
        y_t = self.Lagrange(t_values, y_points)

        return x_t, y_t

    def evaluate_at_points(self, points: List[float], x_points, y_points) -> dict:
        results = {}

        if self.function:
            original_values = [float(self.function.subs(self.X, x))
                               for x in points]
            results['Original Function'] = original_values

        for approach in self.approaches:
            if "SLE" in approach:
                P = self.interpolation_SLE(x_points, y_points)
                values = [P(x) for x in points]
                results['SLE'] = values

            elif "Lagrange" in approach:
                P = self.Lagrange(x_points, y_points)
                values = [P(x) for x in points]
                results['Lagrange'] = values

            elif "Parametric" in approach:
                P_x, P_y = self.parametric_interpolation(x_points, y_points)
                # Dense sampling for better accuracy
                t_values = np.linspace(0, 1, 1000)
                x_values = np.array([P_x(t) for t in t_values])
                y_values = np.array([P_y(t) for t in t_values])

                values = []
                for x in points:
                    idx = np.argmin(np.abs(x_values - x))
                    values.append(y_values[idx])
                results['Parametric'] = values

        return results

    def plot(self, evaluation_points) -> None:
        x_points, y_points = self.generator_points()

        plt.plot(list(x_points), list(y_points), "ok",
                 label="Origin points", zorder=3)

        a, b = self.interval
        x_range = np.linspace(a, b, 200)
        t_value = np.linspace(a, b, 200)

        plt.plot(x_range, [self.function.subs(self.X, x_range[i])
                 for i in range(len(x_range))], "b--", label="original function")

        for approach in self.approaches:
            print(approach)
            if "SLE" in approach:
                P = self.interpolation_SLE(x_points, y_points)
                y_values = [P(x_subs) for x_subs in x_range]
                plt.plot(x_range, y_values, color="orange",
                         label="SLE approach")

            elif "Lagrange" in approach:
                P = self.Lagrange(x_points, y_points)
                y_values = [P(x_subs) for x_subs in x_range]
                plt.plot(x_range, y_values, color="green",
                         label="Lagrange approach")

            elif "Parametric" in approach:
                P_x, P_y = self.parametric_interpolation(x_points, y_points)
                x_values = [P_x(t_i) for t_i in t_value]
                y_values = [P_y(t_i) for t_i in t_value]

                plt.plot(x_values, y_values, color="brown",
                         label="Parametric approach")

        if evaluation_points:
            results = self.evaluate_at_points(
                evaluation_points, x_points, y_points)
            print("\nEvaluation Results:")
            for x in evaluation_points:
                print(f"\nAt x = {x}:")
                for method, values in results.items():
                    idx = evaluation_points.index(x)
                    print(f"{method}: {values[idx]:.6f}")

            if 'Original Function' in results:
                plt.plot(evaluation_points, results['Original Function'], 'k*',
                         markersize=10, label='Evaluation points', zorder=4)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Interpolation Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    ui = UserInterface()
    ui.get_user_input()
    ui.display_user_input()

    function, interval, degree, file_path, approaches, evaluation_points = ui.return_user_input()
    engine = Interpolation(
        function=function,
        interval=interval,
        degree=degree,
        file_path=file_path,
        approaches=approaches
    )

    engine.plot(evaluation_points)
