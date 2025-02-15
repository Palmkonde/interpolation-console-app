import re
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union


class UserInterface:
    def __init__(self) -> None:
        self.function = None
        self.interval = None
        self.degree = None
        self.file_path = None
        self.approaches = []

    def get_function_input(self) -> None:
        self.function = input(
            "Enter a function of a single variable (e.g., x*sin(x) - x^2 + 1): ")

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

    def display_user_input(self) -> None:
        print("\nUser Input:")
        if self.function:
            print(f"Function: {self.function}")
            print(f"Interval: {self.interval}")
            print(f"Polynomial Degree: {self.degree}")
        if self.file_path:
            print(f"File Path: {self.file_path}")
        print(f"Selected Approaches: {', '.join(self.approaches)}")

    def return_user_input(self) -> Tuple[str, 
                                         Tuple[float, float], 
                                         int, 
                                         Union[str, None], 
                                         List[str]]:
        return self.function, self.interval, self.degree, self.file_path, self.approaches

class Interpolation:
    pass

if __name__ == "__main__":
    ui = UserInterface()
    ui.get_user_input()
    ui.display_user_input()