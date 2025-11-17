import numpy as np
import scipy.linalg as la

# Problem 1 - Boundary Value Problems

# Part (a)
N = 1000
A1 = np.pi / N
x = np.linspace(0.0, np.pi, N + 1).reshape((N + 1, 1))
A2 = x.copy()

print("A1 = {}".format(A1))
print("A2 = {}".format(A2))

# Part (b)
b = np.zeros((N + 1, 1), dtype=float)
b[0, 0] = 3.0
b[N, 0] = -3.0

b[1:N, 0] = 4.0 * np.cos(x[1:N, 0]) + 2.0 * np.sin(x[1:N, 0])
A3 = b.copy()
print("A3 = {}".format(A3))

# Part (c)
rowN = np.zeros((1, N + 1), dtype=float)
rowN[0, N] = 1.0

A4 = rowN.copy()
print("A4 = {}".format(A4))

# Part (d)
row3 = np.zeros((1, N + 1), dtype=float)
dx = A1
coef_left  = 1.0 / (dx ** 2) + 1.0 / (2.0 * dx)
coef_mid   = 2.0 - 2.0 / (dx ** 2)
coef_right = 1.0 / (dx ** 2) - 1.0 / (2.0 * dx)
k = 3
row3[0, k - 1] = coef_left
row3[0, k    ] = coef_mid
row3[0, k + 1] = coef_right
A5 = row3.copy()
print("A5 = {}".format(A5))

# Part (e)
L = np.zeros((N + 1, N + 1), dtype=float)
L[0, 0] = 1.0

for k in range(1, N):
    L[k, k - 1] = coef_left
    L[k, k    ] = coef_mid
    L[k, k + 1] = coef_right
L[N, N] = 1.0
A6 = L.copy()
print("A6 = {}".format(A6))

# Part (f)
u = la.solve(L, b)
A7 = u.copy()
print("A7 = {}".format(A7))

# Part (g)
u_exact = 3.0 * np.cos(x) - np.sin(x)
error = np.abs(u_exact - u)
A8 = np.max(error)
print("A8 = {}".format(A8))

# Problem 2 - More Systems of Differential Equations

# System:
# x'(t) = -x + 3y + 1
# y'(t) = -x - 2y + 2
# u(t) = [x(t); y(t)]

# Part (a)
A = np.array([[-1.0,  3.0],
              [-1.0, -2.0]])
b_vec = np.array([[1.0],
                  [2.0]])
A9 = A.copy()
A10 = b_vec.copy()

print("A9 = {}".format(A9))
print("A10 = {}".format(A10))

def f(u, A = A, b = b_vec):
    return A @ u + b
T = 1.0
dt = 0.01
nSteps = int(T / dt)
u0 = np.array([[0.0],
               [0.0]])

# Part (b)
u_fe = u0.copy()
for _ in range(nSteps):
    u_fe = u_fe + dt * f(u_fe)
A11 = u_fe.copy()
print("A11 = {}".format(A11))

# Part (c)
# (I - dt * A) u_{k+1} = u_k + dt * b
I2 = np.eye(2)
M_be = I2 - dt * A

u_be = u0.copy()
for _ in range(nSteps):
    rhs = u_be + dt * b_vec
    u_be = la.solve(M_be, rhs)
A12 = u_be.copy()
print("A12 = {}".format(A12))

# Part (d)
u_exp_trap = u0.copy()
for _ in range(nSteps):
    f_u = f(u_exp_trap)
    u_star = u_exp_trap + dt * f_u
    f_star = f(u_star)
    u_exp_trap = u_exp_trap + 0.5 * dt * (f_u + f_star)
A13 = u_exp_trap.copy()
print("A13 = {}".format(A13))

# Part (e)
# (I - dt/2 * A) u_{k+1} = (I + dt/2 * A) u_k + dt * b
M_it = I2 - 0.5 * dt * A
N_it = I2 + 0.5 * dt * A

u_imp_trap = u0.copy()
for _ in range(nSteps):
    rhs = N_it @ u_imp_trap + dt * b_vec
    u_imp_trap = la.solve(M_it, rhs)
A14 = u_imp_trap.copy()
print("A14 = {}".format(A14))

# Problem 3

# Part (a)
P = np.array([[0.8, 0.1, 0.1],
              [0.3, 0.4, 0.3],
              [0.1, 0.2, 0.7]])
A15 = P.copy()
print("A15 = {}".format(A15))

# Part (b)
p0_rain = np.array([[0.0, 0.0, 1.0]])
p1 = p0_rain @ P
A16 = p1.copy()
print("A16 = {}".format(A16))

# Part (c)
p0_NYE = np.array([[0.8, 0.0, 0.2]])
p_NYD = p0_NYE @ P
A17 = p_NYD.copy()
print("A17 = {}".format(A17))

# Part (d)
steps = 365
p = p0_NYE.copy()
for _ in range(steps):
    p = p @ P
A18 = p.copy()
print("A18 = {}".format(A18))

# Part (e)
p0_cloudy = np.array([[0.0, 1.0, 0.0]])
p_cloudy = p0_cloudy.copy()
for _ in range(steps):
    p_cloudy = p_cloudy @ P
A19 = p_cloudy.copy()
print("A19 = {}".format(A19))

# Part (f)
A20 = np.abs(A18 - A19)
print("A20 = {}".format(A20))
