import argparse
import sys
from sympy import Eq
from .core import LaplaceOperator

def parse_initial_conditions(ic_list):
    ics = {}
    for ic_str in ic_list or []:
        key, val = ic_str.split("=")
        # e.g. "f(0)" or "f'(0)"
        if "f(" in key:
            # strip f( and ) → maybe handle primes
            order = key.count("'")
        else:
            order = 0
        ics[order] = float(val)
    return ics

def main():
    p = argparse.ArgumentParser(prog="LaplaPy",
        description="CLI for LaplaceOperator")
    p.add_argument("expr", help="Function or equation string")
    p.add_argument("--deriv", "-d", type=int, metavar="N",
                   help="Compute Nth derivative")
    p.add_argument("--laplace",   action="store_true",
                   help="Compute Laplace transform")
    p.add_argument("--inverse",   action="store_true",
                   help="Compute inverse Laplace transform")
    p.add_argument("--ode",       action="store_true",
                   help="Solve ODE")
    p.add_argument("--ic",        nargs="+",
                   help="Initial conditions, e.g. f(0)=0 f'(0)=1")
    p.add_argument("--causal",    dest="causal", action="store_true",
                   help="Assume causal (default)")
    p.add_argument("--noncausal", dest="causal", action="store_false",
                   help="Do not assume causal")
    p.add_argument("--quiet",     dest="show_steps", action="store_false",
                   help="Suppress step-by-step output")
    p.set_defaults(causal=True, show_steps=True)

    args = p.parse_args()
    ics = parse_initial_conditions(args.ic)

    op = LaplaceOperator(args.expr,
                         causal=args.causal,
                         show_steps=args.show_steps)

    if args.deriv:
        der = op.derivative(order=args.deriv,
                            initial_conditions=ics)
        print(f"\nDerivative (order {args.deriv}):\n{der}")

    if args.laplace:
        L, roc, poles, zeros = op.laplace()
        print(f"\nLaplace Transform:\n{L}\nROC: {roc}\nPoles: {poles}\nZeros: {zeros}")

    if args.inverse:
        inv = op.inverse_laplace()
        print(f"\nInverse Laplace:\n{inv}")

    if args.ode:
        # user must supply equation of form Eq(...)
        # we rely on sympy to parse it via sympify in the user code
        eq = sympify(args.expr, locals={})
        sol = op.solve_ode(eq, initial_conditions=ics)
        print(f"\nODE Solution:\n{sol}")

if __name__ == "__main__":
    main()
