# LaplaPy: Advanced Symbolic Laplace Transform Analysis

**Scientific Computing Package for Differential Equations, System Analysis, and Control Theory**  
A comprehensive Python library for symbolic Laplace transforms with rigorous mathematical foundations, designed for engineers, scientists, and researchers.

---

## Overview

`LaplaPy` provides a powerful symbolic computation environment for:

1. **Time-domain analysis**: Derivatives, integrals, and function manipulation
2. **Laplace transforms**: With rigorous Region of Convergence (ROC) determination
3. **System analysis**: Pole-zero identification, stability analysis, and frequency response
4. **ODE solving**: Complete solution of linear differential equations with initial conditions
5. **Control system tools**: Bode plots, time-domain responses, and transfer function analysis

---

## Key Features

- **Mathematical Rigor**: Implements Laplace transform theory with proper ROC analysis
- **Causal System Modeling**: Automatic handling of Heaviside functions for physical systems
- **Step-by-Step Solutions**: Educational mode for learning complex concepts
- **Comprehensive System Analysis**: Pole-zero identification, stability criteria, frequency response
- **ODE Solver**: Complete solution workflow for linear differential equations
- **Visualization Tools**: Bode plot generation and time-domain simulations

---

## Installation

```bash
pip install LaplaPy
```

For development:

```bash
git clone https://github.com/4211421036/LaplaPy.git
cd LaplaPy
pip install -e .[dev]
```

---

## Quickstart

### Basic Operations

```python
from LaplaPy import LaplaceOperator, t, s

# Initialize with expression (causal system by default)
op = LaplaceOperator("exp(-3*t) + sin(2*t)", show_steps=True)

# Compute derivative
d1 = op.derivative(order=1)

# Laplace transform with ROC analysis
F_s, roc, poles, zeros = op.laplace()

# Inverse Laplace transform
f_t = op.inverse_laplace()
```

### ODE Solving

```python
from sympy import Eq, Function, Derivative, exp

# Define a differential equation
f = Function('f')(t)
ode = Eq(Derivative(f, t, t) + 3*Derivative(f, t) + 2*f, exp(-t))

# Solve with initial conditions
solution = op.solve_ode(ode, {0: 0, 1: 1})  # f(0)=0, f'(0)=1
```

### System Analysis

```python
# Frequency response
magnitude, phase = op.frequency_response()

# Time-domain response to input
response = op.time_domain_response("sin(4*t)")

# Generate Bode plot data
omega, mag_db, phase_deg = op.bode_plot(ω_range=(0.1, 100), points=100)
```

---

## CLI Usage

```bash
LaplaPy "exp(-2*t)*sin(3*t)" --laplace --deriv 2
LaplaPy "s/(s**2 + 4)" --inverse
LaplaPy "Derivative(f(t), t, t) + 4*f(t) = exp(-t)" --ode --ic "f(0)=0" "f'(0)=1"
```

**Options**:
- `--deriv N`: Compute Nth derivative
- `--laplace`: Compute Laplace transform
- `--inverse`: Compute inverse Laplace transform
- `--ode`: Solve ODE (provide equation)
- `--ic`: Initial conditions (e.g., "f(0)=0", "f'(0)=1")
- `--causal/--noncausal`: System causality assumption
- `--quiet`: Suppress step-by-step output

---

## Mathematical Foundations

### Laplace Transform
$$\mathcal{L}\{f(t)\}(s) = \int_{0^-}^{\infty} e^{-st} f(t)  dt$$

### Derivative Property
$$\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - \sum_{k=1}^{n} s^{n-k} f^{(k-1)}(0^+)$$

### Region of Convergence
- For causal systems: Re(s) > σ_max (right-half plane)
- Proper ROC determination for stability analysis

### Pole-Zero Analysis
- Transfer function: $H(s) = \frac{N(s)}{D(s)}$
- Poles: Roots of denominator polynomial
- Zeros: Roots of numerator polynomial

### Frequency Response
$$H(j\omega) = H(s)\big|_{s=j\omega} = |H(j\omega)| e^{j\angle H(j\omega)}$$

---

## Examples

### Second-Order System Analysis

```python
op = LaplaceOperator("1/(s**2 + 0.6*s + 1)", show_steps=True)

# Get poles and zeros
F_s, roc, poles, zeros = op.laplace()

# Frequency response
magnitude, phase = op.frequency_response()

# Bode plot data
omega, mag_db, phase_deg = op.bode_plot(ω_range=(0.1, 10), points=200)
```

### Circuit Analysis (RLC Network)

```python
# Define circuit equation: L*di/dt + R*i + 1/C*∫i dt = V_in
L, R, C = 0.5, 4, 0.25
op = LaplaceOperator("V_in(s)", show_steps=True)

# Impedance representation
Z = L*s + R + 1/(C*s)
current = op.time_domain_response("V_in(s)/" + str(Z))

# Response to step input
step_response = current.subs("V_in(s)", "1/s")
```

---

## Development & Testing

```bash
# Run tests
pytest tests/

# Generate documentation
cd docs
make html

# Contribution guidelines
CONTRIBUTING.md
```

---

## Scientific Applications

1. **Control Systems**: Stability analysis, controller design
2. **Circuit Analysis**: RLC networks, filter design
3. **Vibration Engineering**: Damped oscillator analysis
4. **Signal Processing**: System response characterization
5. **Communication Systems**: Filter design, modulation analysis
6. **Mechanical Systems**: Spring-mass-damper modeling

---

## Documentation Wiki

Full documentation available at:  
[LaplaPy Documentation WiKi](https://github.com/4211421036/LaplaPy/wiki)

Includes:
- Mathematical background
- API reference
- Tutorial notebooks
- Application examples

---

## License

MIT License

---

## Cite This Work

```bibtex
@software{LaplaPy,
  author = {GALIH RIDHO UTOMO},
  title = {LaplaPy: Advanced Symbolic Laplace Transform Analysis},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/4211421036/LaplaPy}}
}
```
