"""
LaplaPy: A Python package for symbolic differentiation and Laplace transforms with step-by-step output.

Provides:
  - LaplaceOperator: class to parse an expression f(t), compute its derivatives,
    and perform Laplace transforms while printing each intermediate step.
  - convenience symbols t, s for time- and frequency-domain usage.
"""

from .core import LaplaceOperator
from sympy import symbols

# Commonly used symbols
t, s = symbols('t s')

__all__ = ['LaplaceOperator', 't', 's']
