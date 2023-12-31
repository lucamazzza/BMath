import math

from Numerical import root_finding

if __name__ == "__main__":
    print("Bracket of x**2+x: ", root_finding.bracket_of(lambda x: 0.5 * x ** 2 - 1))
    print("Bracket of x**2+x: ", root_finding.bracket_of(lambda x: x**2 - x))

