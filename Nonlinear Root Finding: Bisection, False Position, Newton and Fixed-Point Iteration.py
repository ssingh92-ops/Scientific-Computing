import numpy as np

# Problem 1

f = lambda x: x ** 3 - (6 / 5) * x ** 2 - (9 / 5) * x + (1 / 2)

# part a
A1 = f(-2)
A2 = f(3)
print("A1 = {}".format(A1))
print("A2 = {}".format(A2))

# part b
def bisection1(f, a, b, tol=1e-15, max_steps=10_000):

    if f(a) * f(b) < 0:
        guesses = 1
        for k in range(max_steps):
            x = (a + b) / 2
            if f(x) == 0:
                break
            if np.sign(f(x)) == np.sign(f(a)):
                a = x
            else:
                b = x

            guesses += 1
            if (b - a) < tol:
                break
    else:
        print("there is not root")

    x = (a + b) / 2
    interval = b - a
    return x, guesses, interval


result, iterations, interval = bisection1(f, -2, 3, tol = 1e-10)
A3 = result
print("A3 = {}".format(A3))

# part c
result, iterations, interval = bisection1(f, 0, 1, tol = 1e-10)
A4 = result
print("A4 = {}".format(A4))


# part d
def falsePosition(f, a, b, tol = 1e-10, max_steps = 1000):
    if f(a) * f(b) < 0:
        for k in range(max_steps):
            x = b - f(b) * ((b - a) / (f(b) - f(a)))
            if f(x) == 0:
                print("answer found")
                break

            elif np.sign(f(x)) == np.sign(f(a)):
                a = x
            else:
                b = x

            if b - a < tol:
                print("we got close enough")
                break
    else:
        print('there is not root')
    return x

A5 = falsePosition(f, -2, 3)
print("A5 = {}".format(A5))

# problem 2

def newtonsMethod(f, fprime, x0 = np.pi, tol=1e-15, max_steps=10_000):
    x = x0
    guesses = 1
    fx = f(x)
    if abs(fx) <= tol:
        return x, guesses

    for _ in range(max_steps):
        dfx = fprime(x)
        x = x - fx / dfx
        guesses += 1
        fx = f(x)
        if abs(fx) <= tol:
            break
    return x, guesses

# part a
g = lambda x: x
gprime = lambda x: 1
A6, A7 = newtonsMethod(g, gprime)
print("A6 = {}".format(A6))
print("A7 = {}".format(A7))

# part b
A8, iterations, interval = bisection1(g, -2, np.pi, tol=1e-15)
A9 = iterations
print("A8 = {}".format(A8))
print("A9 = {}".format(A9))

# part c
h = lambda x: x ** 2
hprime = lambda x: 2 * x
A10, A11 = newtonsMethod(h, hprime)
print("A10 = {}".format(A10))
print("A11 = {}".format(A11))

# part d
i = lambda x: x ** 51
iprime = lambda x: 51 * x ** 50
A12, A13 = newtonsMethod(i, iprime)
print("A12 = {}".format(A12))
print("A13 = {}".format(A13))

# part e



A14, iterations, interval = bisection1(i, -2, np.pi, tol=1e-15)
A15 = iterations
print("A14 = {}".format(A14))
print("A15 = {}".format(A15))
print("interval = {}".format(interval))


# problem 3

# part a
g = lambda x: np.sin(x) + 0.5

A16 = np.zeros(100)
A16[0] = 0.0
for k in range(99):
    A16[k+1] = g(A16[k])

# part b

def iterativeMethod(g, x0, max_steps = 10_000 ):
    x = x0
    guesses = 1
    for _ in range(max_steps):
        if np.abs(x - g(x)) < 1e-8:
            x = g(x)
            break
        x = g(x)
        guesses += 1
    return x, guesses

A17, A18 = iterativeMethod(g, 0)
print("A17 = {}".format(A17))
print("A18 = {}".format(A18))

# part c
A19, A20 = iterativeMethod(g, 2025)
print("A19 = {}".format(A19))
print("A20 = {}".format(A20))
