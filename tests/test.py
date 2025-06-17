import pytest
from LaplaPy import LaplaceOperator, t, s
from sympy import exp, sin, cos, Function, Derivative, Eq, pretty

import pytest
from LaplaPy import LaplaceOperator, t, s
from sympy import exp, sin, cos, Function, Derivative, Eq


def test_forward_laplace_basic():
    """
    Test forward Laplace of e^{-3t} + sin(2t).
    L{e^{-3t}} = 1/(s+3), L{sin(2t)} = 2/(s^2+4)
    """
    op = LaplaceOperator(exp(-3*t) + sin(2*t), show_steps=False)
    F = op.forward_laplace()

    expected = 1/(s + 3) + 2/(s**2 + 4)
    assert pytest.approx(F.simplify(), rel=1e-6) == expected.simplify()

    # Check region of convergence and pole-zero lists
    assert "Re(s) > -3" in op.roc or "Re(s) >" in op.roc
    assert all(p in op.poles for p in [ -3 ])
    assert any(z==0 for z in op.zeros)


def test_inverse_laplace_basic():
    """
    Forward + inverse should recover original function.
    """
    f_orig = exp(-3*t) + sin(2*t)
    op = LaplaceOperator(f_orig, show_steps=False)
    F = op.forward_laplace()
    f_recov = op.inverse_laplace(F).simplify()
    assert f_recov.expand() == f_orig.expand()


def test_solve_ode_with_initial_conditions():
    """
    Solve y'' + 3y' + 2y = e^{-t}, y(0)=0, y'(0)=1
    Known solution: y(t) = (e^{-t} - e^{-2t})
    """
    f = Function('f')(t)
    ode = Eq(Derivative(f, t, t) + 3*Derivative(f, t) + 2*f, exp(-t))
    # map sympy objects to IC values
    ics = {f.subs(t,0): 0, Derivative(f, t).subs(t,0): 1}
    op = LaplaceOperator(0, show_steps=False)
    sol = op.solve_ode(ode, ics).simplify()

    # Test initial conditions
    assert sol.subs(t,0) == 0
    assert sol.diff(t).subs(t,0) == 1
    # At large t solution decays
    assert sol.subs(t,10).evalf() == pytest.approx((exp(-10) - exp(-20)).evalf(), rel=1e-6)


def test_system_analysis_properties():
    """
    Analyze H(s) = (s+1)/(s^2 + 0.2*s + 1)
    Check that poles and zeros match, and stability is determined.
    """
    H = (s + 1)/(s**2 + 0.2*s + 1)
    op = LaplaceOperator(H, show_steps=False)
    op.forward_laplace()  # populate poles/zeros
    analysis = op.system_analysis()

    # Check keys
    assert set(analysis.keys()) >= { 'poles', 'zeros', 'stability', 'system_type' }
    # Poles are roots of s^2+0.2s+1
    poles = analysis['poles']
    assert len(poles) == 2
    # Zero at s = -1
    zeros = analysis['zeros']
    assert -1 in zeros
    # Stability: all real parts negative
    assert analysis['stability'] == 'Stable'
    # For this transfer function, no poles at origin => type 0
    assert analysis['system_type'] == 0


def test_bode_plot_returns_correct_lengths():
    """
    Bode plot for a second-order system returns arrays of correct length.
    """
    op = LaplaceOperator(1/(s**2 + 0.5*s + 1), show_steps=False)
    w, mag_db, phase_deg = op.bode_plot(w_min=0.1, w_max=100, points=100)
    assert len(w) == 100
    assert len(mag_db) == 100
    assert len(phase_deg) == 100
    # Frequencies should be within specified range
    assert pytest.approx(w[0], rel=1e-6) == 0.1
    assert pytest.approx(w[-1], rel=1e-6) == 100


def test_time_domain_response_to_sine_input():
    """
    Compute time-domain response of H(s)=1/(s^2+2s+5) to sin(2t).
    """
    H = 1/(s**2 + 2*s + 5)
    op = LaplaceOperator(H, show_steps=False)
    resp = op.time_domain_response(sin(2*t))
    # Response should be an expression in t containing exponentials or sin/cos
    assert resp.free_symbols and t in resp.free_symbols
    assert resp.has(exp) or resp.has(sin) or resp.has(cos)

if __name__ == "__main__":
    test_forward_laplace_basic()
    test_inverse_laplace_basic()
    test_solve_ode_with_initial_conditions()
    test_system_analysis_properties()
    test_bode_plot_returns_correct_lengths()
    test_time_domain_response_to_sine_input()
    print("\nAll tests completed successfully!")
