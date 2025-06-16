"""
LaplaPy: A Python package for symbolic differentiation and Laplace transforms with step-by-step output.

Provides:
  - LaplaceOperator: class to parse an expression f(t), compute its derivatives,
    and perform Laplace transforms while printing each intermediate step.
  - Convenience symbols t, s for time- and frequency-domain usage.
  - Support for derivatives, Laplace transforms, and inverse Laplace transforms.
  - Region of convergence (ROC) information for Laplace transforms.

Features:
  - Step-by-step computation visualization
  - Symbolic computation using SymPy
  - Type checking and validation
  - Clean mathematical output

Example usage:
    >>> from LaplaPy import LaplaceOperator, t, s
    >>> op = LaplaceOperator("exp(-3*t) + sin(2*t)")
    >>> derivative = op.derivative(order=2)
    >>> laplace = op.laplace()
    >>> inverse = op.inverse_laplace()
"""

from .core import LaplaceOperator
from sympy import symbols

# Commonly used symbols
t, s = symbols('t s')

__all__ = ['LaplaceOperator', 't', 's']

# Package version
__version__ = '0.1.2'
