import sys
from sympy import symbols, Function, diff, laplace_transform, sympify, pretty
from sympy.core.expr import Expr

t, s = symbols('t s')

class LaplaceOperator:
    def __init__(self, expr):
        """
        expr: sympy.Expr atau string yang dapat di-sympify,
        mewakili f(t).
        """
        if isinstance(expr, Expr):
            self.expr = expr
        else:
            self.expr = sympify(expr, locals={'t': t})
        print(f"[INIT] f(t) =\n{pretty(self.expr)}\n")

    def derivative(self, order=1, show_steps=True):
        """
        Menghitung d^order/dt^order [f(t)].
        Jika show_steps=True, tampilkan langkah per-langkah.
        """
        if order < 1:
            raise ValueError("Order must be >= 1")
        curr = self.expr
        for i in range(1, order+1):
            der = diff(curr, t)
            if show_steps:
                print(f"[DERIVATIVE Step {i}] d/dt of:\n{pretty(curr)}\n => {pretty(der)}\n")
            curr = der
        return curr

    def laplace(self, expr=None, show_steps=True):
        """
        Menghitung L{ expr }(s). Jika expr=None, pakai self.expr.
        show_steps=True menampilkan hasil sebelum/sesudah simplify.
        """
        target = self.expr if expr is None else expr
        raw = laplace_transform(target, t, s, noconds=False)  # dapat juga conditions
        L, a, cond = raw
        if show_steps:
            print(f"[LAPLACE] Transform {pretty(target)}:")
            print(f"  • Raw: {pretty(L)} , region: Re(s) > {a}")
            print(f"  • Condition: {cond}\n")
        return L

    def laplace_of_derivative(self, order=1, show_steps=True):
        """
        Gabungkan derivative() + laplace() tampil langkah lengkap.
        """
        der = self.derivative(order, show_steps=show_steps)
        return self.laplace(der, show_steps=show_steps)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute derivative & Laplace steps")
    parser.add_argument("expr", help="fungsi f(t), misal exp(-3*t) + t**2")
    parser.add_argument("-d", "--deriv", type=int, default=0,
                        help="hitung turunan orde-N")
    parser.add_argument("-l", "--laplace", action="store_true",
                        help="hitung Laplace f(t)")
    args = parser.parse_args()

    op = LaplaceOperator(args.expr)
    if args.deriv > 0:
        op.derivative(order=args.deriv, show_steps=True)
    if args.laplace:
        op.laplace(show_steps=True)

if __name__ == "__main__":
    main()
