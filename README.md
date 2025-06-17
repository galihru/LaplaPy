# LaplaPy: Advanced Symbolic Laplace Transform Analysis

[![PyPI Version](https://img.shields.io/pypi/v/LaplaPy?color=blue)](https://pypi.org/project/LaplaPy/)
[![PyPI Downloads](https://static.pepy.tech/badge/LaplaPy/month)](https://pepy.tech/project/LaplaPy)
[![GitHub Actions](https://github.com/4211421036/LaplaPy/actions/workflows/py.yml/badge.svg)](https://github.com/4211421036/LaplaPy/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

**LaplaPy** is a professional-grade Python library for:

* **Symbolic Laplace transforms** with rigorous *Region of Convergence* (ROC) analysis
* **Linear ODE solving** via Laplace methods, including initial conditions
* **Control system analysis**: pole-zero maps, stability checks, frequency/time responses
* **Educational, step-by-step output modes** for teaching and self-study

Designed for engineers, scientists, and researchers in control theory, signal processing, electrical/mechanical systems, and applied mathematics.

---

## Features

1. **Forward & Inverse Laplace** with automatic ROC determination
2. **ODE Solver**: direct transform–solve–invert workflow for linear constant-coefficient equations
3. **Pole-Zero Analysis**: identify all poles and zeros symbolically
4. **Stability Assessment**: evaluate pole locations in the complex plane
5. **Frequency Response & Bode**: generate magnitude/phase plots data for `ω ∈ [ω_min, ω_max]`
6. **Time-Domain Response**: compute response to arbitrary inputs (e.g., steps, impulses, sinusoids)
7. **Causal/Non-Causal Modes**: model physical vs. mathematical systems
8. **Step-by-Step Tracing**: verbose mode to print every algebraic step (educational)

---

## Installation

**From PyPI**:

```bash
pip install LaplaPy
```

**From source (dev)**:

```bash
git clone https://github.com/4211421036/LaplaPy.git
cd LaplaPy
pip install -e .[dev]
```

Dependencies: `sympy>=1.10`, `numpy`.

---

## Quickstart Examples

### 1. Basic Laplace Transform

```python
from LaplaPy import LaplaceOperator, t, s

# f(t) = e^{-3t} + sin(2t)
op = LaplaceOperator("exp(-3*t) + sin(2*t)", show_steps=False)
F_s = op.forward_laplace()
print("F(s) =", F_s)       # 1/(s+3) + 2/(s^2+4)
print("ROC:", op.roc)       # Re(s) > -3
```

### 2. Inverse Transform

```python
# Recover f(t)
f_recov = op.inverse_laplace()
print("f(t) =", f_recov)    # exp(-3*t) + sin(2*t)
```

### 3. Solve ODE

```python
from sympy import Eq, Function, Derivative, exp

f = Function('f')(t)
ode = Eq(Derivative(f, t, 2) + 3*Derivative(f, t) + 2*f,
         exp(-t))
# ICs: f(0)=0, f'(0)=1
sol = LaplaceOperator(0).solve_ode(node,
      {f.subs(t,0):0, Derivative(f,t).subs(t,0):1})
print(sol)  # (e^{-t} - e^{-2t})
```

### 4. System Analysis & Bode Data

```python
# H(s) = (s+1)/(s^2+0.2*s+1)
op = LaplaceOperator("(s+1)/(s**2+0.2*s+1)")
op.forward_laplace()
analysis = op.system_analysis()
# poles, zeros, stability
defp print(analysis)
# Bode data
ω, mag_db, phase = op.bode_plot(w_min=0.1, w_max=100, points=200)
```

### 5. Time-Domain Response

```python
# Response to sin(4t)
r = op.time_domain_response("sin(4*t)")
print(r)
```

---

## CLI Usage

The `LaplaPy` console script provides a quick interface:

```bash
# Laplace transform + 2nd derivative:
LaplaPy "exp(-2*t)*sin(3*t)" --laplace

# Inverse Laplace:
LaplaPy "1/(s**2+4)" --inverse

# Solve ODE with ICs:
LaplaPy "f''(t)+4*f(t)=exp(-t)" --ode \
        --ic "f(0)=0" "f'(0)=1"
```

**Flags**:

* `--laplace` (`-L`) : forward transform
* `--inverse` (`-I`): inverse transform
* `--ode` (`-O`)    : solve ODE
* `--ic`: initial conditions
* `--quiet`: suppress verbose steps
* `--causal/--noncausal`: choose system causality

---

## 📚 Mathematical Foundations

### Laplace Transform Definition

$$
\mathcal{L}\{f(t)\}(s)
= \int_{0^-}^{\infty} e^{-st}f(t)\,dt
$$

### Derivative Property

$$
\mathcal{L}\{f^{(n)}(t)\}(s)
= s^nF(s)-\sum_{k=0}^{n-1}s^{n-1-k}f^{(k)}(0^+)
$$

### Region of Convergence (ROC)

* For causal signals: \$\mathrm{Re}(s) > \max(\mathrm{Re}(\text{poles}))\$
* ROC ensures transform integrals converge and stability criteria

### Pole-Zero & Stability

* **Poles**: roots of denominator \$D(s)=0\$
* **Zeros**: roots of numerator \$N(s)=0\$
* **Stability**: all poles in left-half complex plane

### Frequency Response

$$
H(j\omega)=H(s)\big|_{s=j\omega}
=|H(j\omega)|e^{j\angle H(j\omega)}
$$

---

## Development & Testing

```bash
# Install dev extras
pip install -e .[dev]

# Run test suite
pytest -q

# Lint with ruff
ruff .

# Type-check (mypy)
mypy LaplaPy
```

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{LaplaPy,
  author    = {GALIH RIDHO UTOMO},
  title     = {LaplaPy: Advanced Symbolic Laplace Transform Analysis},
  year      = {2025},
  url       = {https://github.com/4211421036/LaplaPy},
  version   = {0.2.2}
}
```
