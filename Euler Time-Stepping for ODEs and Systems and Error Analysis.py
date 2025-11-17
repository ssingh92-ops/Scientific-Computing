import numpy as np


# Problem 1
xPrime = lambda x, t: - 0.1 * x - (1 + (0.1 ** 2) / 4) * np.exp((-0.1 * t) / 2) * np.sin(t)
xExact = lambda t: np.exp((-0.1 * t) / 2) * (np.cos(t) - (0.1 / 2) * np.sin(t))
x0 = 1


def forwardEuler(x0, T = 20, dt = 0.1, f = xPrime):
    N = int(T / dt) + 1
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    x[0] = x0

    for i in range(N - 1):
        x[i + 1] = x[i] + dt * f(x[i], t[i])

    return x[-1]

# Part a
T = 20
dt = 0.1
A1 = forwardEuler(x0, T, dt, xPrime)
A2 = abs(A1 - xExact(20))
print("A1 = {}".format(A1))
print("A2 = {}".format(A2))

# Part b
dt1 = 0.01
A3 = forwardEuler(x0, T, dt1, xPrime)
A4 = abs(A3 - xExact(20))
print("A3 = {}".format(A3))
print("A4 = {}".format(A4))

def backwardEuler(x0, T = 20, dt = 0.1):
    a = 0.1
    r = lambda t: -(1 + (a ** 2) / 4) * np.exp((-a * t) / 2) * np.sin(t)
    denom = 1 + a * dt

    N = int(T / dt) + 1
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    x[0] = x0

    for i in range(N - 1):
        x[i+1] = (x[i] + dt * r(t[i + 1])) / denom

    return x[-1]

# Part c
A5 = backwardEuler(x0, T, dt)
A6 = abs(A5 - xExact(20))
print("A5 = {}".format(A5))
print("A6 = {}".format(A6))

# Part d
A7 = backwardEuler(x0, T, dt1)
A8 = abs(A7 - xExact(20))
print("A7 = {}".format(A7))
print("A8 = {}".format(A8))

# Problem 2
yPrime = lambda x: 8 * np.sin(x)
yExact = lambda t: 2 * np.arctan(np.exp(8 * t) / (1 + np.sqrt(2)))
x02 = np.pi / 4

def forwardEuler2(x0, T = 20, dt = 0.1, f = xPrime):
    N = int(T / dt) + 1
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    x[0] = x0

    for i in range(N - 1):
        x[i + 1] = x[i] + dt * f(x[i])

    return t, x

# Part a
T1 = 2
t1, x1 = forwardEuler2(x02, T1, dt1, yPrime)
A9 = float(x1[100])
print("A9 = {}".format(A9))

# Part b
def maxError(t, x, exactFxn = yExact):
    err = np.abs(x - exactFxn(t))
    return float(np.max(err))
A10 = maxError(t1, x1)
print("A10 = {}".format(A10))

# Part c
dt2 = 0.001
t2, x2 = forwardEuler2(x02, T1, dt2, yPrime)
A11 = float(x2[1000])
print("A11 = {}".format(A11))

# Part d
A12 = maxError(t2, x2)
print("A12 = {}".format(A12))

# Part e
A13 = A10/A12
print("A13 = {}".format(A13))

# Part f
def backwardEuler2_bisect(x0, T = 2.0, dt = 0.1, tol = 1e-12, max_iter = 200):
    N = int(T/dt) + 1
    t = np.linspace(0.0, T, N)
    x = np.zeros(N); x[0] = x0

    for k in range(N-1):
        def f(z):
            return z - x[k] - 8.0 * dt * np.sin(z)

        a, b = 0.0, 1.5 * np.pi
        fa, fb = f(a), f(b)

        for _ in range(max_iter):
            m = 0.5*(a + b)
            fm = f(m)
            if (b - a)/2.0 <= tol:
                x[k+1] = m
                break
            if fa*fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        else:
            x[k+1] = 0.5*(a + b)

    return t, x

t3, x3 = backwardEuler2_bisect(x02, T=T1, dt=0.1, tol=1e-12)
A14 = float(x3[10])
print("A14 = {}".format(A14))

# Part g
A15 = maxError(t3, x3)
print("A15 = {}".format(A15))

# Problem 3
xprime1 = lambda x, y: 2 * x - 2 * y
yprime1 = lambda x: x
x0x = 1
x0y = -1

# Part a
def feSys(dt=0.1, T=4.0, x0=1.0, y0=-1.0):
    N = int(T/dt) + 1
    x = np.zeros(N); y = np.zeros(N)
    x[0], y[0] = x0, y0
    for k in range(N-1):
        x[k+1] = x[k] + dt*(2*x[k] - 2*y[k])
        y[k+1] = y[k] + dt*x[k]
    return x, y

dt3 = 0.1; T = 4.0
xFe, yFe = feSys(dt3, T)
A16 = float(xFe[-1])   # x(4)
A17 = float(yFe[-1])   # y(4)
print("A16 = {}".format(A16))
print("A17 = {}".format(A17))

# Part b
def be_sys(dt=0.1, T=4.0, x0=1.0, y0=-1.0):
    N = int(T/dt) + 1
    x = np.zeros(N); y = np.zeros(N)
    x[0], y[0] = x0, y0
    denom = 1 - 2*dt + 2*(dt**2)
    for k in range(N-1):
        x[k + 1] = (x[k] - 2*dt*y[k]) / denom
        y[k + 1] = y[k] + dt*x[k + 1]
    return x, y

x_be, y_be = be_sys(dt, T)
A18 = float(x_be[-1])
A19 = float(y_be[-1])
print("A18 = {}".format(A18))
print("A19 = {}".format(A19))

# Part c
A20 = np.abs(A17 - A19)
print("A20 = {}".format(A20))
