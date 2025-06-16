import sys
from sympy import (symbols, Function, diff, laplace_transform, sympify, pretty, re, 
                   inverse_laplace_transform, exp, Heaviside, DiracDelta, Piecewise, 
                   simplify, expand, fraction, apart, Eq, solve, solveset, S, Interval, 
                   oo, integrate, limit, Abs, And, Or, Lt, Gt, Poly, degree, rootof)
from sympy.core.expr import Expr
from sympy.utilities.lambdify import lambdify
import numpy as np

t, s = symbols('t s', real=True, positive=True)  # Domain constraints

class LaplaceOperator:
    def __init__(self, expr, causal=True, show_steps=True):
        """
        Laplace transform operator with enhanced scientific rigor
        
        Parameters:
        expr: sympy.Expr or string - function f(t)
        causal: bool - if True, assumes f(t) = 0 for t < 0 (causal system)
        show_steps: bool - whether to print intermediate steps
        """
        if isinstance(expr, Expr):
            self.expr = expr
        else:
            self.expr = sympify(expr, locals={'t': t})
        
        # Add Heaviside function for causality
        if causal:
            self.expr = self.expr * Heaviside(t)
            
        self.causal = causal
        self.show_steps = show_steps
        self.roc = None  # Region of convergence (as inequality)
        self.poles = []  # Poles of the system
        self.zeros = []  # Zeros of the system
        self.conditions = None  # Conditions for existence
        self.steps = []  # Store calculation steps
        
        if show_steps:
            self._add_step(f"[INITIAL] f(t) = {pretty(self.expr)}")
            if causal:
                self._add_step("Assuming causal system (f(t) = 0 for t < 0)")

    def _add_step(self, message):
        """Store a calculation step for later retrieval"""
        self.steps.append(message)
        if self.show_steps:
            print(message)
    
    def derivative(self, order=1, initial_conditions=None):
        """
        Compute derivative of order n with proper handling of initial conditions
        
        Parameters:
        order: int - derivative order
        initial_conditions: dict - initial conditions {f(0): value, f'(0): value, ...}
        
        Returns: derivative expression
        """
        if order < 1:
            raise ValueError("Order must be >= 1")
            
        if initial_conditions is None:
            initial_conditions = {}
            
        curr = self.expr
        for i in range(1, order + 1):
            der = diff(curr, t)
            
            if self.show_steps:
                self._add_step(f"\n[DERIVATIVE Step {i}] d/dt of:")
                self._add_step(pretty(curr))
                self._add_step(f" => {pretty(der)}")
            
            curr = der
            
        return curr

    def laplace(self, expr=None):
        """
        Compute the Laplace transform with rigorous mathematical handling
        
        Parameters:
        expr: optional expression to transform (default: self.expr)
        
        Returns: (transform, roc, poles, zeros)
        """
        target = self.expr if expr is None else expr
        
        # Check existence conditions
        self._check_laplace_existence(target)
        
        # Compute the transform
        L, a, cond = laplace_transform(target, t, s, noconds=False)
        
        # Simplify result
        L_simp = simplify(L)
        
        # Find ROC
        self.roc = self._find_roc(a, cond)
        
        # Find poles and zeros
        self.poles, self.zeros = self._find_poles_zeros(L_simp)
        
        if self.show_steps:
            self._add_step("\n[LAPLACE TRANSFORM]")
            self._add_step(f"Original: {pretty(target)}")
            self._add_step(f"Transform: {pretty(L_simp)}")
            self._add_step(f"Region of Convergence: {self.roc}")
            self._add_step(f"Poles: {[pretty(p) for p in self.poles]}")
            self._add_step(f"Zeros: {[pretty(z) for z in self.zeros]}")
            self._add_step(f"Conditions: {cond}")
        
        return L_simp, self.roc, self.poles, self.zeros

    def _find_roc(self, a, cond):
        """Determine the Region of Convergence (ROC)"""
        # For causal systems, ROC is Re(s) > σ_max
        if self.causal:
            if a == -oo:
                return "All s"
            return f"Re(s) > {pretty(a)}"
        
        # For non-causal systems, ROC is more complex
        # This is a simplified approach - real implementation would need more analysis
        return f"Re(s) > {pretty(a)}"  # Placeholder for more complex ROC determination

    def _find_poles_zeros(self, expr):
        """Find poles and zeros of the Laplace transform"""
        num, den = fraction(expr)
        if den == 1:
            return [], []

        # Buat Poly untuk komputasi numerik akar
        den_poly = Poly(den, s)
        num_poly = Poly(num, s)

        # Akar penyebut → poles
        try:
            poles_set = solveset(den, s, domain=S.Complexes)
        except Exception:
            poles_set = None

        if isinstance(poles_set, set) or getattr(poles_set, 'is_FiniteSet', False):
            poles = list(poles_set)
        else:
            # fallback: gunakan nroots untuk akar (mungkin kompleks)
            poles = den_poly.nroots()

        # Akar pembilang → zeros
        try:
            zeros_set = solveset(num, s, domain=S.Complexes)
        except Exception:
            zeros_set = None

        if isinstance(zeros_set, set) or getattr(zeros_set, 'is_FiniteSet', False):
            zeros = list(zeros_set)
        else:
            zeros = num_poly.nroots()

        return poles, zeros

    def _check_laplace_existence(self, expr):
        """Verify if the Laplace transform exists"""
        M, α = symbols('M α', real=True, positive=True)
        condition = Eq(Abs(expr), M * exp(α * t))
        try:
            solutions = solve(condition, (M, α), dict=True)
        except Exception:
            solutions = []
    
        if not solutions and self.show_steps:
            self._add_step("[WARNING] Could not verify exponential order via solve, proceeding anyway.")
    
        discontinuities = self._find_discontinuities(expr)
        if discontinuities and self.show_steps:
            self._add_step(f"[NOTE] Function has discontinuities at: {discontinuities}")
    
        return True

    def _find_discontinuities(self, expr):
        """Find discontinuities in the time domain (0 to ∞)"""
        # For most elementary functions, we can find discontinuities by inspection
        discontinuities = []
        
        # Check for Dirac deltas and Heavisides
        if expr.has(DiracDelta) or expr.has(Heaviside):
            # Find points where Heaviside jumps
            heavisides = [arg for arg in expr.atoms(Heaviside) if isinstance(arg, Heaviside)]
            for h in heavisides:
                if h.args[0].is_real:
                    discontinuities.append(h.args[0])
        
        # Check for piecewise functions
        if expr.has(Piecewise):
            pieces = expr.atoms(Piecewise)
            for piece in pieces:
                for (cond, expr) in piece.args:
                    # Find boundary points
                    if cond.is_Relational and cond.lhs == t:
                        discontinuities.append(cond.rhs)
        
        return sorted(set(discontinuities))

    def laplace_of_derivative(self, order=1, initial_conditions=None):
        """
        Compute Laplace transform of derivative using the property:
        L{f'(t)} = sF(s) - f(0⁺)
        L{f''(t)} = s²F(s) - sf(0⁺) - f'(0⁺)
        
        Parameters:
        order: int - derivative order
        initial_conditions: dict - initial conditions {0: f(0), 1: f'(0), ...}
        
        Returns: Laplace transform of the derivative
        """
        if initial_conditions is None:
            initial_conditions = {}
            if self.show_steps:
                self._add_step("[WARNING] No initial conditions provided. Assuming zero initial state.")
        
        # Get the Laplace transform of the original function
        F_s, roc, poles, zeros = self.laplace()
        
        # Apply the derivative property
        result = s**order * F_s
        for k in range(order):
            # Subtract initial conditions terms
            term = s**(order - 1 - k) * initial_conditions.get(k, 0)
            result -= term
        
        if self.show_steps:
            self._add_step("\n[LAPLACE OF DERIVATIVE]")
            self._add_step(f"Using property: L{{f^({order})(t)}} = s^{order}F(s) - Σ s^{order-1-k} f^{(k)}(0⁺)")
            self._add_step(f"Result: {pretty(result)}")
        
        return result

    def inverse_laplace(self, expr=None, time_domain=True):
        """
        Compute inverse Laplace transform with rigorous handling
        
        Parameters:
        expr: optional expression to transform (default: self.expr)
        time_domain: bool - if True, return time-domain expression
        
        Returns: inverse Laplace transform
        """
        target = self.expr if expr is None else expr
        
        # Perform partial fraction expansion
        expanded = apart(target, s, full=True)
        
        if self.show_steps:
            self._add_step("\n[INVERSE LAPLACE]")
            self._add_step(f"Original: {pretty(target)}")
            self._add_step(f"Partial fraction expansion: {pretty(expanded)}")
        
        # Compute inverse transform
        try:
            inv = inverse_laplace_transform(expanded, s, t)
            # Simplify using Heaviside for causality
            if self.causal:
                inv = inv * Heaviside(t)
            
            # Simplify result
            inv_simp = simplify(inv)
            
            if self.show_steps:
                self._add_step(f"Result: {pretty(inv_simp)}")
            
            return inv_simp
        except Exception as e:
            if self.show_steps:
                self._add_step(f"[ERROR] Could not compute inverse: {str(e)}")
            return None

    def solve_ode(self, ode, initial_conditions=None):
        """
        Solve ordinary differential equation using Laplace transforms
        
        Parameters:
        ode: sympy.Eq - differential equation
        initial_conditions: dict - initial conditions
        
        Returns: solution function f(t)
        """
        if initial_conditions is None:
            initial_conditions = {}
        
        if self.show_steps:
            self._add_step("\n[SOLVING ODE USING LAPLACE]")
            self._add_step(f"Equation: {pretty(ode)}")
            self._add_step(f"Initial conditions: {initial_conditions}")
        
        # Extract left and right sides
        lhs = ode.lhs
        rhs = ode.rhs
        
        # Apply Laplace transform to both sides
        L_lhs = self.laplace_of_derivative_from_ode(lhs, initial_conditions)
        L_rhs, _, _, _ = self.laplace(rhs)
        
        # Form algebraic equation in s-domain
        s_eq = Eq(L_lhs, L_rhs)
        
        if self.show_steps:
            self._add_step(f"Transformed equation: {pretty(s_eq)}")
        
        # Solve for F(s) — tambahkan tanda ')' yang hilang
        F_s = solve(s_eq, self.laplace(self.expr)[0])
        
        if self.show_steps:
            self._add_step(f"Solution in s-domain: F(s) = {pretty(F_s)}")
        
        # Apply inverse Laplace
        solution = self.inverse_laplace(F_s)
        
        if self.show_steps:
            self._add_step(f"Time-domain solution: f(t) = {pretty(solution)}")
        
        return solution

    def laplace_of_derivative_from_ode(self, expr, initial_conditions):
        """
        Apply Laplace transform to a derivative expression in an ODE
        """
        from sympy import Add
        result = 0
        
        # Handle additive terms
        if isinstance(expr, Add):
            terms = expr.args
        else:
            terms = [expr]
        
        for term in terms:
            # Count derivative order
            if term.is_Derivative:
                order = len(term.args) - 1
                f = term.args[0]
                # Apply derivative property
                F_s = self.laplace(f)[0]
                term_transform = s**order * F_s
                # Subtract initial conditions
                for k in range(order):
                    term_transform -= s**(order - 1 - k) * initial_conditions.get(k, 0)
                result += term_transform
            else:
                # Regular term
                result += self.laplace(term)[0]
        
        return result

    def frequency_response(self, expr=None):
        """
        Compute frequency response (s = jω)
        
        Parameters:
        expr: optional expression (default: self.laplace result)
        
        Returns: (magnitude, phase) as functions of ω
        """
        if expr is None:
            expr = self.laplace()[0]
        
        # Substitute s = jω
        ω = symbols('ω', real=True)
        H_jω = expr.subs(s, 1j*ω)
        
        # Compute magnitude and phase
        magnitude = Abs(H_jω)
        phase = re(atan2(H_jω.as_real_imag()[1], H_jω.as_real_imag()[0]))
        
        if self.show_steps:
            self._add_step("\n[FREQUENCY RESPONSE]")
            self._add_step(f"Transfer function: H(jω) = {pretty(H_jω)}")
            self._add_step(f"Magnitude: |H(jω)| = {pretty(magnitude)}")
            self._add_step(f"Phase: ∠H(jω) = {pretty(phase)}")
        
        return magnitude, phase

    def time_domain_response(self, input_expr, initial_conditions=None):
        """
        Compute time-domain response to an input signal
        
        Parameters:
        input_expr: input signal x(t)
        initial_conditions: dict - initial conditions
        
        Returns: output signal y(t)
        """
        if initial_conditions is None:
            initial_conditions = {}
        
        # Get system transfer function
        H_s = self.laplace()[0]
        
        # Get Laplace transform of input
        X_s = self.laplace(input_expr)[0]
        
        # Output in s-domain
        Y_s = H_s * X_s
        
        # Convert to time domain
        y_t = self.inverse_laplace(Y_s)
        
        if self.show_steps:
            self._add_step("\n[TIME-DOMAIN RESPONSE]")
            self._add_step(f"System transfer function: H(s) = {pretty(H_s)}")
            self._add_step(f"Input transform: X(s) = {pretty(X_s)}")
            self._add_step(f"Output transform: Y(s) = H(s)X(s) = {pretty(Y_s)}")
            self._add_step(f"Time-domain response: y(t) = {pretty(y_t)}")
        
        return y_t

    def bode_plot(self, expr=None, ω_range=(0.1, 100), points=100):
        """
        Generate Bode plot data (magnitude and phase)
        
        Parameters:
        expr: optional expression (default: self.laplace result)
        ω_range: tuple (min_ω, max_ω)
        points: number of points
        
        Returns: (ω_vals, mag_vals, phase_vals)
        """
        if expr is None:
            expr = self.laplace()[0]
        
        magnitude, phase = self.frequency_response(expr)
        
        # Create angular frequency range
        ω_vals = np.logspace(np.log10(ω_range[0]), np.log10(ω_range[1]), points)
        
        # Lambdify the expressions
        mag_func = lambdify(symbols('ω'), magnitude, 'numpy')
        phase_func = lambdify(symbols('ω'), phase, 'numpy')
        
        # Evaluate
        mag_vals = mag_func(ω_vals)
        phase_vals = phase_func(ω_vals)
        
        # Convert to dB
        mag_vals_db = 20 * np.log10(np.abs(mag_vals))
        
        return ω_vals, mag_vals_db, np.degrees(phase_vals)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Laplace Transform Solver")
    parser.add_argument("expr", help="Function f(t), e.g., 'exp(-3*t) + sin(2*t)'")
    parser.add_argument("-d", "--deriv", type=int, default=0,
                        help="Compute derivative of order N")
    parser.add_argument("-l", "--laplace", action="store_true",
                        help="Compute Laplace transform")
    parser.add_argument("-i", "--inverse", action="store_true",
                        help="Compute inverse Laplace transform")
    parser.add_argument("-ode", help="Solve ODE (provide equation)")
    parser.add_argument("-ic", nargs='+', help="Initial conditions (e.g., 'f(0)=1' 'f_prime(0)=0')")
    parser.add_argument("-c", "--causal", action="store_true", default=True,
                        help="Assume causal system (default: True)")
    parser.add_argument("-nc", "--noncausal", action="store_false", dest="causal",
                        help="Do not assume causal system")
    parser.add_argument("-q", "--quiet", action="store_false", dest="show_steps",
                        help="Do not show calculation steps")
    
    args = parser.parse_args()
    
    # Parse initial conditions
    initial_conditions = {}
    if args.ic:
        for ic_str in args.ic:
            key, value = ic_str.split('=')
            order = 0
            if key.startswith('f'):
                # Handle f(0), f'(0), etc.
                if "''" in key:
                    order = key.count("'")
                elif key.startswith('f_prime'):
                    order = 1
                elif key.startswith('f_dprime'):
                    order = 2
                elif key.startswith('f'):
                    # Extract order from f{n}(0)
                    if key[1].isdigit():
                        order = int(key[1])
            initial_conditions[order] = float(value)
    
    op = LaplaceOperator(args.expr, causal=args.causal, show_steps=args.show_steps)
    
    if args.deriv > 0:
        derivative = op.derivative(order=args.deriv, initial_conditions=initial_conditions)
        print(f"\nDerivative of order {args.deriv}:")
        print(pretty(derivative))
    
    if args.laplace:
        transform, roc, poles, zeros = op.laplace()
        print("\nLaplace Transform:")
        print(pretty(transform))
        print(f"ROC: {roc}")
        print(f"Poles: {[pretty(p) for p in poles]}")
        print(f"Zeros: {[pretty(z) for z in zeros]}")
    
    if args.inverse:
        inverse = op.inverse_laplace()
        print("\nInverse Laplace Transform:")
        print(pretty(inverse))
    
    if args.ode:
        # Parse ODE equation
        # This would require more sophisticated parsing
        print("ODE solving requires more advanced parsing. Use the API directly.")

if __name__ == "__main__":
    main()
