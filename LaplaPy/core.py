from sympy import symbols, Function, diff, laplace_transform

# deklarasi simbol
t, s = symbols('t s')
f = Function('f')

def derivative(expr, order=1):
    """
    Mengambil turunan orde `order` terhadap t
    `expr` bisa berupa sympy expression fungsi f(t), 
    misal expr = f(t) atau t**2*exp(-t)
    """
    return diff(expr, (t, order))

def laplace(expr):
    """
    Menghitung L{ expr(t) }(s) tanpa kondisi (noconds=True)
    """
    L, _, _ = laplace_transform(expr, t, s, noconds=True)
    return L

def laplace_of_derivative(expr, order=1):
    """
    Langsung menghitung L{ d^order/dt^order [expr(t)] }
    """
    der = derivative(expr, order)
    return laplace(der)
