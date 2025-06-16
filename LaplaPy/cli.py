import argparse
import sys
from sympy import Eq, sympify, pretty
from .core import LaplaceOperator

def parse_initial_conditions(ic_list):
    """Parse list like ['f(0)=1', \"f'(0)=0\"] → {0:1.0,1:0.0}"""
    ics = {}
    for ic_str in ic_list or []:
        key, val = ic_str.split("=")
        # count number of primes for derivative order
        order = key.count("'")
        ics[order] = float(val)
    return ics

def main():
    parser = argparse.ArgumentParser(prog="LaplaPy",
        description="CLI for LaplaceOperator")
    parser.add_argument("expr", help="Function f(t) or ODE string")
    parser.add_argument("--deriv", "-d", type=int, metavar="N",
                        help="Compute Nth derivative of f(t)")
    parser.add_argument("--laplace",   action="store_true",
                        help="Compute Laplace transform")
    parser.add_argument("--inverse",   action="store_true",
                        help="Compute inverse Laplace transform")
    parser.add_argument("--ode",       action="store_true",
                        help="Solve ODE given as Eq(lhs, rhs)")
    parser.add_argument("--ic",        nargs="+",
                        help="Initial conditions, e.g. f(0)=0 f'(0)=1")
    parser.add_argument("--causal",    dest="causal", action="store_true",
                        help="Assume causal (default)")
    parser.add_argument("--noncausal", dest="causal", action="store_false",
                        help="Do not assume causal")
    parser.add_argument("--quiet",     dest="show_steps", action="store_false",
                        help="Suppress step‐by‐step output")
    parser.set_defaults(causal=True, show_steps=True)

    args = parser.parse_args()
    ics = parse_initial_conditions(args.ic)

    expr = args.expr
    if args.ode:
        try:
            expr = sympify(expr, locals={})
        except Exception as e:
            print(f"Error parsing ODE: {e}", file=sys.stderr)
            sys.exit(1)

    op = LaplaceOperator(expr,
                         causal=args.causal,
                         show_steps=args.show_steps)

    if args.deriv:
        try:
            der = op.derivative(order=args.deriv,
                                initial_conditions=ics)
            print(f"\nDerivative (order {args.deriv}):\n{pretty(der)}")
        except Exception as e:
            print(f"Error computing derivative: {e}", file=sys.stderr)
            if args.show_steps: raise

    if args.laplace:
        try:
            L, roc, poles, zeros = op.laplace()
            print(f"\nLaplace Transform:\n{pretty(L)}")
            print(f"ROC: {roc}")
            print(f"Poles: {[pretty(p) for p in poles]}")
            print(f"Zeros: {[pretty(z) for z in zeros]}")
        except ValueError as e:
            print(f"Warning: Could not compute poles/zeros automatically: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error computing Laplace: {e}", file=sys.stderr)
            if args.show_steps: raise

    if args.inverse:
        try:
            inv = op.inverse_laplace()
            print(f"\nInverse Laplace:\n{pretty(inv)}")
        except Exception as e:
            print(f"Error computing inverse Laplace: {e}", file=sys.stderr)
            if args.show_steps: raise

    # 4) Solve ODE
    if args.ode:
        try:
            sol = op.solve_ode(expr, initial_conditions=ics)
            print(f"\nODE Solution:\n{pretty(sol)}")
        except Exception as e:
            print(f"Error solving ODE: {e}", file=sys.stderr)
            if args.show_steps: raise

if __name__ == "__main__":
    main()
