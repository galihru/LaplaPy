import pytest
from laplaceop.core import LaplaceOperator
from sympy import exp

def test_derivative():
    op = LaplaceOperator("t**3")
    d2 = op.derivative(order=2, show_steps=False)
    assert str(d2) == "6*t"

def test_laplace():
    op = LaplaceOperator(exp(-3*t))
    L = op.laplace(show_steps=False)
    assert str(L) == "1/(s + 3)"

def test_laplace_of_derivative():
    op = LaplaceOperator(exp(-3*t))
    Ld = op.laplace_of_derivative(order=1, show_steps=False)
    # L{d/dt e^{-3t}} = s/(s+3) - 1
    assert "s/(s + 3)" in str(Ld)
