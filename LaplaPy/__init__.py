"""
LaplaPy: Advanced Python package for symbolic Laplace transforms with rigorous mathematical foundations.

Provides:
  - LaplaceOperator: Class for symbolic analysis of differential equations using Laplace transforms
  - Symbols t, s: Time and complex frequency domain variables
  - Comprehensive Laplace transform operations with ROC analysis
  - ODE solving capabilities with initial conditions
  - System analysis tools (frequency response, Bode plots)

Features:
  - Causal system modeling (f(t) = 0 for t < 0)
  - Region of Convergence (ROC) determination
  - Pole-zero analysis for stability determination
  - Partial fraction expansion for inverse transforms
  - Step-by-step computation visualization
  - Bode plot generation for frequency analysis
  - Time-domain response simulation

Example usage:
    >>> from LaplaPy import LaplaceOperator, t, s
    >>> op = LaplaceOperator("exp(-3*t) + sin(2*t)")
    
    # Compute derivatives
    >>> d1 = op.derivative(order=1)
    
    # Laplace transform with ROC analysis
    >>> F_s, roc, poles, zeros = op.laplace()
    
    # Solve ODE with initial conditions
    >>> from sympy import Eq, Function, Derivative
    >>> f = Function('f')(t)
    >>> ode = Eq(Derivative(f, t, t) + 3*Derivative(f, t) + 2*f, exp(-t))
    >>> solution = op.solve_ode(ode, {0: 0, 1: 1})
    
    # Frequency response analysis
    >>> magnitude, phase = op.frequency_response()
    
    # Time-domain response to input
    >>> response = op.time_domain_response("sin(4*t)")
    
    # Bode plot data
    >>> omega, mag_db, phase_deg = op.bode_plot()

Version: 1.0.0 (Scientific Edition)
"""

from .core import LaplaceOperator
from sympy import symbols

# Time and complex frequency domain symbols
t, s = symbols('t s', real=True, positive=True)

__all__ = ['LaplaceOperator', 't', 's']

# Package version
__version__ = '0.2.1'
