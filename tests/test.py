import pytest
from LaplaPy import LaplaceOperator, t, s
from sympy import exp, sin, cos, Function, Derivative, Eq, pretty

def test_basic_operations():
    print("\n=== Basic Operations Test ===")
    # Initialize with causal assumption (default)
    op = LaplaceOperator("exp(-3*t) + sin(2*t)", show_steps=True)
    
    # Compute derivatives
    d1 = op.derivative(order=1)
    print(f"First derivative: {pretty(d1)}")
    
    # Laplace transform
    F_s, roc, poles, zeros = op.laplace()
    print(f"Laplace transform: {pretty(F_s)}")
    print(f"ROC: {roc}")
    print(f"Poles: {[pretty(p) for p in poles]}")
    print(f"Zeros: {[pretty(z) for z in zeros]}")
    
    # Inverse Laplace
    inv = op.inverse_laplace(F_s)
    print(f"Inverse Laplace: {pretty(inv)}")

def test_ode_solving():
    print("\n=== ODE Solving Test ===")
    op = LaplaceOperator("f(t)", show_steps=True)
    
    # Define ODE: y'' + 3y' + 2y = e^{-t} with initial conditions
    f = Function('f')(t)
    ode = Eq(Derivative(f, t, t) + 3*Derivative(f, t) + 2*f, exp(-t))
    
    # Initial conditions: y(0) = 0, y'(0) = 1
    solution = op.solve_ode(ode, {0: 0, 1: 1})
    print(f"ODE Solution: {pretty(solution)}")

def test_system_analysis():
    print("\n=== System Analysis Test ===")
    # Define a system: H(s) = 1/(s^2 + 2s + 5)
    op = LaplaceOperator("1/(s**2 + 2*s + 5)", show_steps=False)
    
    # Frequency response
    magnitude, phase = op.frequency_response()
    print(f"Magnitude response: {pretty(magnitude)}")
    print(f"Phase response: {pretty(phase)}")
    
    # Time-domain response to input
    response = op.time_domain_response("sin(2*t)")
    print(f"Time-domain response: {pretty(response)}")

def test_bode_plot():
    print("\n=== Bode Plot Test ===")
    # Create a second-order system
    op = LaplaceOperator("1/(s**2 + 0.5*s + 1)", show_steps=False)
    
    # Generate Bode plot data
    omega, mag_db, phase_deg = op.bode_plot(ω_range=(0.1, 100), points=50)
    
    print(f"Bode plot data generated:")
    print(f"ω range: {omega[0]:.2f} to {omega[-1]:.2f} rad/s")
    print(f"Magnitude range: {min(mag_db):.2f} dB to {max(mag_db):.2f} dB")
    print(f"Phase range: {min(phase_deg):.2f}° to {max(phase_deg):.2f}°")

def test_advanced_features():
    print("\n=== Advanced Features Test ===")
    # Non-causal system
    op = LaplaceOperator("exp(2*t)", causal=False, show_steps=True)
    
    # Laplace transform of non-causal function
    F_s, roc, poles, zeros = op.laplace()
    print(f"Non-causal Laplace transform: {pretty(F_s)}")
    print(f"ROC: {roc}")
    
    # Derivative with initial conditions
    derivative = op.derivative(order=1, initial_conditions={0: 1})
    print(f"Derivative with IC: {pretty(derivative)}")
    
    # Laplace of derivative with IC
    L_deriv = op.laplace_of_derivative(order=1, initial_conditions={0: 1})
    print(f"Laplace of derivative: {pretty(L_deriv)}")

if __name__ == "__main__":
    test_basic_operations()
    test_ode_solving()
    test_system_analysis()
    test_bode_plot()
    test_advanced_features()
    print("\nAll tests completed successfully!")
