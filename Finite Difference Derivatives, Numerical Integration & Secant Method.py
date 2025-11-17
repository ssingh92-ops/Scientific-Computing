import numpy as np
import pandas as pd

# Problem 1

# Part a
x0 = np.pi / 4
f = lambda x: x * np.sin(x)
fprimeX0 =  (1 / np.sqrt(2)) + (np.pi / (4 * np.sqrt(2)))
fprime2X0 = np.sqrt(2) - (np.pi / (4 * np.sqrt(2)))

A1 = fprimeX0
print("A1 = {}".format(A1))

# Part b
def forward_diff_scheme(f, x, dx):
    return (f(x + dx) - f(x)) / dx

def backward_diff_scheme(f, x, dx):
    return (f(x) - f(x - dx)) / dx

def central_diff_scheme(f, x, dx):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

d1 = forward_diff_scheme(f, x0, .01)
d2 = forward_diff_scheme(f, x0, .001)
q = np.abs(d2 - fprimeX0) / np.abs(d1 - fprimeX0)
array = [d1, d2, q]
A2 = np.array(array)
print("A2 = {}".format(A2))

# Part c
d3 = central_diff_scheme(f, x0, .01)
d4 = central_diff_scheme(f, x0, .001)
q1 = np.abs(d4 - fprimeX0) / np.abs(d3 - fprimeX0)
array1 = [d3, d4, q1]
A3 = np.array(array1)
print("A3 = {}".format(A3))

# Part d
partd = lambda x, dx: (-3 * f(x) + 4 * f(x + dx) - f(x + 2 * dx)) / ( 2 * dx)
d5 = partd(x0, .01)
d6 = partd(x0, .001)
q = np.abs(d6 - fprimeX0) / np.abs(d5 - fprimeX0)
array2 = [d5, d6, q]
A4 = np.array(array2)
print("A4 = {}".format(A4))

# Part e
parte = lambda x, dx: (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx ** 2)
A5 = parte(x0, .01)
print("A5 = {}".format(A5))

A6 = np.abs(A5 - fprime2X0)
print("A6 = {}".format(A6))


# Problem 2

# Part a
df = pd.read_csv('population.csv')
t = df['t'].to_numpy()
N = df['N'].to_numpy()

A7 = N[8]
print("A7 = {}".format(A7))

# Part b
dN = t[1] - t[0]
Nprime = np.zeros(t.size)
Nprime[0] = (N[1] - N[0]) / dN
for j in range(1, t.size - 1):
    Nprime[j] = (N[j+1] - N[j-1]) / (2 * dN)
Nprime[-1] = (N[-1] - N[-2]) / dN
A8 = Nprime[8]
print("A8 = {}".format(A8))

# Part c
A9 = Nprime[0]
print("A9 = {}".format(A9))

# Part d
A10 = Nprime[-1]
print("A10 = {}".format(A10))

# Part e
A11 = Nprime
print("A11 = {}".format(A11))

# Part f
LHR = 0.0
for j in range(t.size - 1):
    LHR += dN * N[j]
A12 = LHR
print("A12 = {}".format(A12))

# Part g
RHR = 0.0
for j in range(1, t.size):
    RHR += dN * N[j]
A13 = RHR
print("A13 = {}".format(A13))

# Part f
Trapezoid = 0.0
Trapezoid = Trapezoid + (dN / 2) * N[0] + (dN / 2) * N[-1]
for j in range(1, t.size - 1):
    Trapezoid += dN * N[j]
A14 = Trapezoid
print("A14 = {}".format(A14))


# Problem 3

# Part a and b
h = lambda x: x ** 2 - 2

def secant_method(f, x0, x1):
    guesses = np.zeros(10)
    guesses[0] = x0
    guesses[1] = x1

    absolute_error = np.zeros(10)
    absolute_error[0] = np.abs(x0 - np.sqrt(2))
    absolute_error[1] = np.abs(x1 - np.sqrt(2))
    for j in range(8):
        guesses[j + 2] = guesses[j+1] - ((f(guesses[j + 1]) * (guesses[j + 1] - guesses[j])) / (f(guesses[j + 1]) - f(guesses[j])) )
        absolute_error[j + 2] = np.abs(guesses[j + 2] - np.sqrt(2))
    return guesses, absolute_error

A15, A16 = secant_method(h, 1, 2)
print("A15 = {}".format(A15))
print("A16 = {}".format(A16))

# Part c
def secant_method1(f, x0, x1, tol):
    guesses = 2
    while np.abs(x1 - np.sqrt(2)) > tol:
            y = x1
            x1 = x1 - ((f(x1) * (x1 - x0)) / (f(x1) - f(x0)))
            x0 = y
            guesses += 1
    return guesses

A17 = secant_method1(h, 2025, 2024, 1e-12)
print("A17 = {}".format(A17))

# Part d
def secant_method2(f, x0, x1, tol):
    guesses = 2
    while np.abs(f(x1)) > tol:
            y = x1
            x1 = x1 - ((f(x1) * (x1 - x0)) / (f(x1) - f(x0)))
            x0 = y
            guesses += 1
    return guesses
A18 = secant_method2(h, 2025, 2024, 1e-12)
print("A18 = {}".format(A18))

# Part e
def secant_method3(f, x0, x1, tol):
    guesses = 2
    while np.abs(x1 - x0) > tol:
            y = x1
            x1 = x1 - ((f(x1) * (x1 - x0)) / (f(x1) - f(x0)))
            x0 = y
            guesses += 1
    return x1, guesses
A19, A20 = secant_method3(h, 2025, 2024, 1e-12)
print("A19 = {}".format(A19))
print("A20 = {}".format(A20))
