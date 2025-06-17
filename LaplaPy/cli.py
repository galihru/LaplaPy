import argparse
import sys
from sympy import Eq, sympify, pretty, symbols, Function, Derivative
from .core import LaplaceOperator

t, s = symbols('t s', real=True, positive=True)

def parse_initial_conditions(ic_list, func_symbol):
    """
    Convert strings like f(0)=1, f'(0)=0 into a dict
      { f(0): 1, f'(0): 0 }
    where f = func_symbol(t).
    """
    ics = {}
    for ic_str in ic_list or []:
        left, val = ic_str.split("=")
        val = sympify(val)
        # count primes to get derivative order
        order = left.count("'")
        if order == 0:
            key = func_symbol.subs(t, 0)
        else:
            key = Derivative(func_symbol, t, order).subs(t, 0)
        ics[key] = val
    return ics

def main():
    parser = argparse.ArgumentParser(prog="LaplaPy",
        description="CLI for LaplaceOperator")
    parser.add_argument("expr",
        help="Function f(t), e.g. \"exp(-2*t)*sin(3*t)\" "
             "or ODE \"f''(t)+4*f(t)=exp(-t)\"")
    parser.add_argument("--laplace",   "-L", action="store_true",
                        help="Compute Laplace transform of f(t)")
    parser.add_argument("--inverse",   "-I", action="store_true",
                        help="Compute inverse Laplace transform")
    parser.add_argument("--ode",       "-O", action="store_true",
                        help="Solve ODE given as \"LHS=RHS\"")
    parser.add_argument("--ic",        nargs="+",
                        help="Initial conditions, e.g. f(0)=0 f'(0)=1")
    parser.add_argument("--quiet",     dest="show_steps", action="store_false",
                        help="Suppress step-by-step output")
    parser.set_defaults(show_steps=True)

    args = parser.parse_args()

    # ODE mode
    if args.ode:
        if "=" not in args.expr:
            print("Error: ODE must contain '='", file=sys.stderr)
            sys.exit(1)
        lhs_str, rhs_str = args.expr.split("=", 1)
        try:
            lhs = sympify(lhs_str, locals={'t': t})
            rhs = sympify(rhs_str, locals={'t': t})
            ode_eq = Eq(lhs, rhs)
        except Exception as e:
            print(f"Error parsing ODE: {e}", file=sys.stderr)
            sys.exit(1)

        # assume function is the first Function in LHS
        funcs = list(lhs.atoms(Function))
        if not funcs:
            print("Error: No function f(t) found in ODE", file=sys.stderr)
            sys.exit(1)
        f = funcs[0]
        ics = parse_initial_conditions(args.ic, f)

        op = LaplaceOperator(0, independent_var=t, show_steps=args.show_steps)
        try:
            sol = op.solve_ode(ode_eq, ics)
            print("\nODE solution:")
            print(pretty(sol))
        except Exception as e:
            print(f"Error solving ODE: {e}", file=sys.stderr)
            if args.show_steps:
                raise
        return

    # Non-ODE mode: treat expr as time‐domain function
    try:
        f_expr = sympify(args.expr, locals={'t': t})
    except Exception as e:
        print(f"Error parsing expression: {e}", file=sys.stderr)
        sys.exit(1)

    op = LaplaceOperator(f_expr, independent_var=t, show_steps=args.show_steps)

    if args.laplace:
        try:
            F = op.forward_laplace()
            print("\nLaplace transform F(s):")
            print(pretty(F))
            print("ROC:", op.roc)
            print("Poles:", [pretty(p) for p in op.poles])
            print("Zeros:", [pretty(z) for z in op.zeros])
        except Exception as e:
            print(f"Error computing Laplace: {e}", file=sys.stderr)
            if args.show_steps:
                raise

    if args.inverse:
        try:
            inv = op.inverse_laplace()
            print("\nInverse Laplace f(t):")
            print(pretty(inv))
        except Exception as e:
            print(f"Error computing inverse Laplace: {e}", file=sys.stderr)
            if args.show_steps:
                raise

if __name__ == "__main__":
    main()
