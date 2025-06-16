# LaplaPy

<p align="center" style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.5em;">
  <a href="https://pypi.org/project/LaplaPy/">
    <img src="https://img.shields.io/pypi/v/LaplaPy.svg" alt="PyPI version"/>
  </a>
  <a href="https://pypi.org/project/LaplaPy/">
    <img src="https://img.shields.io/pypi/dm/LaplaPy.svg" alt="PyPI downloads"/>
  </a>
</p>

**Symbolic Differentiation & Laplace Transform Utility**  
A scientific Python library for step-by-step symbolic computation of time-domain derivatives and their Laplace transforms.

---

## Overview

`LaplaPy` enables:

1. **Parsing** of an arbitrary time-domain expression \(f(t)\).  
2. **Symbolic differentiation** up to any integer order.  
3. **Laplace transform** of either the original function or its derivatives, with region-of-convergence details.  
4. **Step-by-step console output** to illustrate each mathematical operation.

---

## Installation

```bash
pip install LaplaPy
````

Or for local development:

```bash
git clone https://github.com/4211421036/LaplaPy.git
cd LaplaPy
pip install -e .
```

---

## Quickstart

```python
from LaplaPy import LaplaceOperator, t, s
from sympy import exp, sin

# 1. Initialize with a symbolic expression
op = LaplaceOperator("exp(-3*t) + sin(2*t)")

# 2. Compute first derivative, showing each step:
d1 = op.derivative(order=1)
# Console prints:
# [DERIVATIVE Step 1] d/dt of exp(-3 t) + sin(2 t)
#   => -3·exp(-3 t) + 2·cos(2 t)

# 3. Compute second derivative:
d2 = op.derivative(order=2)
# Console prints two sequential differentiation steps.

# 4. Compute Laplace transform of the original f(t):
L0 = op.laplace()
# Console prints region of convergence and raw result.

# 5. Compute Laplace transform of the first derivative:
L1 = op.laplace_of_derivative(order=1)
```

---

## CLI Usage

After installation, you can invoke the command-line interface:

```bash
LaplaPy "t**2 * exp(-5*t)" -d 2 -l
```

* `-d N` or `--deriv N` : compute $N$th derivative of $f(t)$.
* `-l` or `--laplace` : compute Laplace transform of $f(t)$.

Output is printed in human-readable, stepwise format.

---

## Mathematical Background

* **Symbolic Differentiation**
  Uses SymPy’s `diff` under the hood to compute

  $$\frac{\mathrm{d}^n}{\mathrm{d}t^n} f(t).$$

* **Laplace Transform**
  Computes

  $$\mathcal{L}\{f(t)\}(s) = \int_{0^-}^{\infty} e^{-s t}\,f(t)\,\mathrm{d}t,$$

  reporting both the transformed expression and its region of convergence.

---

## Example

For $f(t) = t^3 e^{-4t}$:

```bash
$ LaplaPy "t**3*exp(-4*t)" -d 1 -l
[INIT] f(t) =
       3
     t  ⋅ℯ⁻⁴⋅t

[DERIVATIVE Step 1]
d/dt of t³⋅e⁻⁴t
 => 3⋅t²⋅e⁻⁴t − 4⋅t³⋅e⁻⁴t

[LAPLACE]
Transform of t³⋅e⁻⁴t:
  • Raw: 6/(s + 4)⁴ , region: Re(s) > −4
  • ...conditions...
```

---

## Development & Testing

```bash
pip install -r requirements-dev.txt
pytest tests/
```

---

## License

MIT License
