import sys
import numpy as np
from sympy import (symbols, Function, diff, laplace_transform, sympify, 
                   inverse_laplace_transform, exp, Heaviside, DiracDelta, 
                   simplify, fraction, apart, Eq, solve, solveset, S, 
                   Abs, re, im, atan2, I, pi, N, latex, Poly, degree,
                   sin, cos, tan, sinh, cosh, tanh, log, sqrt, factorial,
                   Matrix, zeros, pprint, Eq, solve_linear_system, 
                   integrate, limit, oo, Piecewise, Symbol, Wild, 
                   preorder_traversal, Add, Mul, Float, Rational)
from sympy.core.expr import Expr
from sympy.utilities.lambdify import lambdify
from sympy.integrals.transforms import _fast_inverse_laplace
from sympy.physics.control.lti import TransferFunction
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global symbols
t, s = symbols('t s', real=True, positive=True)
omega = symbols('omega', real=True)

class LaplaceOperator:
    """
    General Laplace Transform Module
    Handles arbitrary functions and systems, not limited to specific domains
    """
    def __init__(self, expr, independent_var=t, show_steps=True):
        """
        Initialize the Laplace transformer
        
        Parameters:
        expr: sympy.Expr or string - function to transform
        independent_var: symbol - independent variable (default: t)
        show_steps: bool - whether to display calculation steps
        """
        self.independent_var = independent_var
        self.show_steps = show_steps
        self.steps = []
        
        # Parse expression
        if isinstance(expr, Expr):
            self.expr = expr
        else:
            self.expr = sympify(expr)
            
        self.original_expr = self.expr
        self.laplace_expr = None
        self.inverse_expr = None
        self.poles = []
        self.zeros = []
        self.roc = ""
        self.region_of_convergence = None
        
        if show_steps:
            self._add_step(f"Initial expression: {self._pretty(self.expr)}")
    
    def _add_step(self, message, expr=None):
        """Add a calculation step"""
        if expr is not None:
            message += f": {self._pretty(expr)}"
        self.steps.append(message)
        if self.show_steps:
            print(message)
    
    def _pretty(self, expr):
        """Format expression nicely"""
        return latex(expr) if self.show_steps else str(expr)
    
    def _forward_transform(self, expr):
        """
        Compute Laplace transform
        F(s) = ∫₀^∞ f(t)e^{-st} dt
        """
        try:
            # Try standard transform first
            F, a, cond = laplace_transform(
                expr, 
                self.independent_var, 
                s, 
                noconds=False
            )
            self.region_of_convergence = f"Re(s) > {a}" if a != -oo else "All s"
            return simplify(F)
        except Exception:
            # Fallback to manual integration for special cases
            self._add_step("[BOAS METHOD] Using direct integration")
            integrand = expr * exp(-s*self.independent_var)
            F = integrate(integrand, (self.independent_var, 0, oo))
            return simplify(F)
    
    def _inverse_transform(self, expr):
        """
        Compute inverse Laplace transform using Boas approach
        f(t) = (1/(2πi)) ∫_{γ-i∞}^{γ+i∞} F(s)e^{st} ds
        """
        try:
            # Try standard inverse first
            f = inverse_laplace_transform(
                expr, 
                s, 
                self.independent_var, 
                noconds=True
            )
            return simplify(f)
        except Exception:
            # Use partial fraction expansion and table lookup
            self._add_step("[BOAS METHOD] Using partial fraction expansion")
            return self._partial_fraction_inverse(expr)
    
    def _partial_fraction_inverse(self, expr):
        """Compute inverse via partial fraction decomposition"""
        # Expand and decompose
        expanded = apart(expr, s, full=True)
        self._add_step("Partial fraction expansion", expanded)
        
        # Inverse each term
        terms = expanded.as_ordered_terms()
        result = 0
        
        for term in terms:
            # Handle different term types
            if term.is_rational_function(s):
                # Rational term - use standard inverse
                inv_term = inverse_laplace_transform(
                    term, s, self.independent_var, noconds=True
                )
            elif term.has(exp):
                # Exponential shift
                inv_term = self._handle_exponential_shift(term)
            else:
                # General case
                inv_term = _fast_inverse_laplace(term, s, self.independent_var)
            
            result += inv_term
        
        return simplify(result)
    
    def _handle_exponential_shift(self, term):
        """Handle exponential shifts in s-domain"""
        # Look for exponential factors: e^{-as} form
        a = Wild('a', exclude=[s])
        match = term.match(exp(-a*s))
        
        if match:
            a_val = match[a]
            # Remove exponential factor
            remaining = term / exp(-a_val*s)
            # Compute inverse of remaining part
            inv_remaining = self._inverse_transform(remaining)
            # Apply time shift
            return inv_remaining.subs(self.independent_var, self.independent_var - a_val)
        else:
            return _fast_inverse_laplace(term, s, self.independent_var)
    
    def _find_poles_zeros(self, expr):
        """Find poles and zeros of a rational function"""
        num, den = fraction(expr)
        num_poly = Poly(num, s)
        den_poly = Poly(den, s)
        
        # Find zeros (roots of numerator)
        try:
            zeros = solveset(num_poly, s, domain=S.Complexes)
        except Exception:
            zeros = []
        
        # Find poles (roots of denominator)
        try:
            poles = solveset(den_poly, s, domain=S.Complexes)
        except Exception:
            poles = []
        
        return poles, zeros
    
    def forward_laplace(self):
        """Compute forward Laplace transform with detailed steps"""
        self._add_step("\n=== FORWARD LAPLACE TRANSFORM ===")
        
        # Step 1: Apply definition
        self._add_step("[Step 1] Apply Laplace transform definition")
        self._add_step(f"F(s) = ∫₀^∞ [{self._pretty(self.expr)}] e^(-s t) dt")
        
        # Step 2: Compute transform
        self.laplace_expr = self._forward_transform(self.expr)
        self._add_step("[Step 2] Computed transform", self.laplace_expr)
        
        # Step 3: Identify poles and zeros
        self.poles, self.zeros = self._find_poles_zeros(self.laplace_expr)
        self._add_step(f"[Step 3] Poles: {self._pretty(self.poles)}")
        self._add_step(f"[Step 3] Zeros: {self._pretty(self.zeros)}")
        
        # Step 4: Region of convergence
        if self.region_of_convergence:
            self.roc = self.region_of_convergence
        else:
            # Basic ROC determination for causal systems
            real_parts = [re(p) for p in self.poles if re(p) != oo]
            if real_parts:
                sigma_max = max(real_parts)
                self.roc = f"Re(s) > {sigma_max}"
            else:
                self.roc = "All s"
        self._add_step(f"[Step 4] Region of Convergence: {self.roc}")
        
        return self.laplace_expr
    
    def inverse_laplace(self, expr=None):
        """Compute inverse Laplace transform with detailed steps"""
        target = expr or self.laplace_expr
        if target is None:
            raise ValueError("No Laplace expression available")
        
        self._add_step("\n=== INVERSE LAPLACE TRANSFORM ===")
        self._add_step(f"[Input] F(s) = {self._pretty(target)}")
        
        # Step 1: Apply definition
        self._add_step("[Step 1] Apply inverse Laplace transform definition")
        self._add_step(f"f(t) = (1/2πi) ∫ F(s)e^(st) ds")
        
        # Step 2: Compute inverse
        self.inverse_expr = self._inverse_transform(target)
        self._add_step("[Step 2] Computed inverse", self.inverse_expr)
        
        # Step 3: Simplify result
        simplified = simplify(self.inverse_expr)
        if simplified != self.inverse_expr:
            self.inverse_expr = simplified
            self._add_step("[Step 3] Simplified result", self.inverse_expr)
        
        return self.inverse_expr
    
    def solve_ode(self, ode, ics=None):
        """
        Solve ODE using Laplace transforms
        Reference: Boas Chapter 8, Section 12
        
        Parameters:
        ode: sympy.Eq - The ODE to solve
        ics: dict - Initial conditions {y(0): val, y'(0): val, ...}
        """
        if not isinstance(ode, Eq):
            raise ValueError("ODE must be an equation")
        
        self._add_step("\n=== SOLVING ODE WITH LAPLACE TRANSFORMS ===")
        self._add_step(f"[ODE] {self._pretty(ode)}")
        
        if ics:
            self._add_step(f"[Initial Conditions] {ics}")
        
        # Step 1: Apply Laplace transform to both sides
        lhs_laplace = self._transform_ode_side(ode.lhs, ics)
        rhs_laplace = self._transform_ode_side(ode.rhs, ics)
        
        self._add_step("[Step 1] Transformed equation:")
        self._add_step(f"LHS: {self._pretty(lhs_laplace)}")
        self._add_step(f"RHS: {self._pretty(rhs_laplace)}")
        
        # Step 2: Solve for Y(s)
        y_sym = Function('Y')(s)
        solution_eq = Eq(lhs_laplace, rhs_laplace)
        y_solution = solve(solution_eq, y_sym)[0]
        self._add_step("[Step 2] Solved for Y(s)", y_solution)
        
        # Step 3: Apply inverse Laplace
        y_t = self.inverse_laplace(y_solution)
        self._add_step("[Step 3] Inverse transform", y_t)
        
        # Step 4: Apply initial conditions if any
        if ics:
            y_t = self._apply_ics(y_t, ics)
            self._add_step("[Step 4] Applied initial conditions", y_t)
        
        return y_t
    
    def _transform_ode_side(self, expr, ics):
        """Transform one side of an ODE using Laplace properties"""
        from sympy import Derivative
        result = 0
        
        # Handle additive terms
        if expr.is_Add:
            terms = expr.args
        else:
            terms = [expr]
        
        for term in terms:
            # Handle derivatives
            if isinstance(term, Derivative):
                f = term.args[0]
                order = len(term.variables)
                
                # Transform derivative
                transform = s**order * Function('Y')(s)
                
                # Subtract initial conditions
                for j in range(order):
                    if ics:
                        # Get initial condition for j-th derivative
                        ics_key = diff(f, self.independent_var, j)
                        ics_val = ics.get(ics_key.subs(self.independent_var, 0), 0)
                        transform -= s**(order-1-j) * ics_val
                    else:
                        transform -= s**(order-1-j) * 0
                
                result += transform
            else:
                # Transform non-derivative term
                result += self._forward_transform(term)
        
        return result
    
    def _apply_ics(self, solution, ics):
        """Apply initial conditions to solution"""
        # Create equations from initial conditions
        equations = []
        derivatives = {0: solution}
        max_deriv = max([diff(func, self.independent_var).count(self.independent_var) 
                        for func in ics.keys()])
        
        # Compute necessary derivatives
        for i in range(1, max_deriv+1):
            derivatives[i] = diff(derivatives[i-1], self.independent_var)
        
        # Create equations
        for func, value in ics.items():
            # Find derivative order
            if func.has(Derivative):
                order = func.variables[0][1]
            else:
                order = 0
            
            # Create equation
            eq = Eq(derivatives[order].subs(self.independent_var, 0), value)
            equations.append(eq)
        
        # Solve system
        constants = solution.free_symbols - {self.independent_var}
        sol = solve(equations, constants)
        
        if sol:
            return solution.subs(sol)
        return solution
    
    def system_analysis(self, tf_expr=None):
        """
        Analyze a general system transfer function
        H(s) = numerator / denominator
        
        Parameters:
        tf_expr: Transfer function expression (optional)
        """
        expr = tf_expr or self.laplace_expr
        if expr is None:
            raise ValueError("No transfer function available")
        
        self._add_step("\n=== SYSTEM ANALYSIS ===")
        self._add_step(f"[Transfer Function] H(s) = {self._pretty(expr)}")
        
        # Step 1: Pole-zero analysis
        poles, zeros = self._find_poles_zeros(expr)
        self._add_step("[Poles]", poles)
        self._add_step("[Zeros]", zeros)
        
        # Step 2: Stability analysis
        stable = True
        for pole in poles:
            if re(pole) > 0:
                stable = False
                break
        stability = "Stable" if stable else "Unstable"
        self._add_step(f"[Stability] {stability}")
        
        # Step 3: Frequency response
        try:
            w = symbols('ω', real=True)
            H_jw = expr.subs(s, I*w)
            magnitude = Abs(H_jw)
            phase = atan2(im(H_jw), re(H_jw))
            
            self._add_step("[Frequency Response]")
            self._add_step(f"|H(jω)| = {self._pretty(magnitude)}")
            self._add_step(f"∠H(jω) = {self._pretty(phase)}")
            
            # Bode plot data
            w_vals = np.logspace(-2, 2, 500)
            mag_func = lambdify(w, magnitude, 'numpy')
            phase_func = lambdify(w, phase, 'numpy')
            
            mag_vals = mag_func(w_vals)
            phase_vals = phase_func(w_vals)
            
            # Find resonant frequency
            peak_idx = np.argmax(mag_vals)
            w_peak = w_vals[peak_idx]
            mag_peak = mag_vals[peak_idx]
            
            self._add_step(f"[Resonance] ω_res = {w_peak:.3f} rad/s, |H|_peak = {mag_peak:.3f}")
        except Exception as e:
            self._add_step(f"[Frequency Analysis Error] {str(e)}")
        
        # Step 4: Time-domain characteristics
        try:
            step_response = self.inverse_laplace(expr/s)
            self._add_step("[Step Response]", step_response)
            
            # Steady-state value
            ss_value = limit(step_response, self.independent_var, oo)
            self._add_step(f"[Steady-State] {ss_value}")
            
            # Rise time (10% to 90%)
            if ss_value != 0 and ss_value != oo:
                t_vals = np.linspace(0, 10, 1000)
                resp_func = lambdify(self.independent_var, step_response, 'numpy')
                resp_vals = resp_func(t_vals)
                
                # Normalize
                resp_vals /= ss_value
                
                # Find indices
                idx_10 = np.argmax(resp_vals > 0.1)
                idx_90 = np.argmax(resp_vals > 0.9)
                
                if idx_10 > 0 and idx_90 > 0:
                    rise_time = t_vals[idx_90] - t_vals[idx_10]
                    self._add_step(f"[Rise Time] t_r = {rise_time:.3f} s (10% to 90%)")
        except Exception as e:
            self._add_step(f"[Time-Domain Analysis Error] {str(e)}")
        
        # Step 5: System type and error constants
        num, den = fraction(expr)
        den_poly = Poly(den, s)
        num_poly = Poly(num, s)
        
        # System type (number of poles at origin)
        type_n = den_poly.coeff_monomial(1)  # s^0 coefficient
        if type_n == 0:
            system_type = degree(den_poly) - degree(den_poly - den_poly.LC()*s**degree(den_poly))
        else:
            system_type = 0
        
        self._add_step(f"[System Type] Type {system_type}")
        
        # Error constants
        if system_type == 0:
            k_p = limit(expr, s, 0)
            self._add_step(f"[Position Error Constant] K_p = {k_p}")
        elif system_type == 1:
            k_v = limit(s*expr, s, 0)
            self._add_step(f"[Velocity Error Constant] K_v = {k_v}")
        elif system_type >= 2:
            k_a = limit(s**2*expr, s, 0)
            self._add_step(f"[Acceleration Error Constant] K_a = {k_a}")
        
        return {
            'poles': poles,
            'zeros': zeros,
            'stability': stability,
            'system_type': system_type
        }

    def bode_plot(self, tf_expr=None, w_min=0.1, w_max=100, points=200):
        """
        Generate Bode plot data for a transfer function
        
        Returns:
        (frequencies, magnitude_dB, phase_degrees)
        """
        expr = tf_expr or self.laplace_expr
        if expr is None:
            raise ValueError("No transfer function available")
        
        w = symbols('ω', real=True)
        H_jw = expr.subs(s, I*w)
        
        magnitude = Abs(H_jw)
        phase = atan2(im(H_jw), re(H_jw))
        
        # Create frequency range
        w_vals = np.logspace(np.log10(w_min), np.log10(w_max), points)
        
        # Lambdify
        mag_func = lambdify(w, magnitude, 'numpy')
        phase_func = lambdify(w, phase, 'numpy')
        
        # Evaluate
        mag_vals = mag_func(w_vals)
        phase_vals = phase_func(w_vals)
        
        # Convert to dB and degrees
        mag_db = 20 * np.log10(np.abs(mag_vals))
        phase_deg = np.degrees(np.unwrap(phase_vals))
        
        return w_vals, mag_db, phase_deg

    def time_domain_response(self, input_expr, ics=None):
        """
        Compute time-domain response to arbitrary input
        
        Parameters:
        input_expr: Input signal in time domain
        ics: Initial conditions for system
        """
        # Get system transfer function
        if self.laplace_expr is None:
            self.forward_laplace()
        H_s = self.laplace_expr
        
        # Transform input
        input_transformer = LaplaceOperator(
            input_expr, 
            independent_var=self.independent_var,
            show_steps=False
        )
        X_s = input_transformer.forward_laplace()
        
        # Output in s-domain
        Y_s = H_s * X_s
        
        # Transform back to time domain
        output_transformer = LaplaceOperator(
            Y_s, 
            independent_var=s,
            show_steps=False
        )
        y_t = output_transformer.inverse_laplace()
        
        # Apply initial conditions if specified
        if ics:
            y_t = self._apply_ics(y_t, ics)
        
        return y_t

    def get_steps(self):
        """Get all calculation steps"""
        return self.steps

    def clear_steps(self):
        """Clear calculation history"""
        self.steps = []


def test_cases():
    """Test various cases"""
    print("\n" + "="*60)
    print(" TEST CASE 1: Exponential Decay (Boas Example)")
    print("="*60)
    case1 = LaplaceOperator("exp(-a*t)", show_steps=True)
    F1 = case1.forward_laplace()
    print("\nResult:", latex(F1))
    
    print("\n" + "="*60)
    print(" TEST CASE 2: Trigonometric Function (Boas Example)")
    print("="*60)
    case2 = LaplaceOperator("sin(omega*t)", show_steps=True)
    F2 = case2.forward_laplace()
    print("\nResult:", latex(F2))
    
    print("\n" + "="*60)
    print(" TEST CASE 3: Second-Order System (General)")
    print("="*60)
    case3 = LaplaceOperator("1/(s**2 + 3*s + 2)", show_steps=True)
    f3 = case3.inverse_laplace()
    print("\nResult:", latex(f3))
    
    print("\n" + "="*60)
    print(" TEST CASE 4: ODE Solution (Boas Chapter 8)")
    print("="*60)
    t = symbols('t')
    y = Function('y')(t)
    ode = Eq(diff(y, t, t) + 3*diff(y, t) + 2*y, exp(-t))
    ics = {y.subs(t, 0): 0, diff(y, t).subs(t, 0): 1}
    case4 = LaplaceOperator(y, show_steps=True)
    solution = case4.solve_ode(ode, ics)
    print("\nSolution:", latex(solution))
    
    print("\n" + "="*60)
    print(" TEST CASE 5: System Analysis (General)")
    print("="*60)
    case5 = LaplaceOperator("(s+1)/(s**2 + 0.2*s + 1)", show_steps=True)
    case5.forward_laplace()
    analysis = case5.system_analysis()
    print("\nAnalysis:")
    pprint(analysis)
    
    print("\n" + "="*60)
    print(" TEST CASE 6: Time-Domain Response (General)")
    print("="*60)
    case6 = LaplaceOperator("1/(s**2 + 2*s + 5)", show_steps=True)
    response = case6.time_domain_response("Heaviside(t)")
    print("\nStep Response:", latex(response))

if __name__ == "__main__":
    test_cases()
