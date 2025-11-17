import numpy as np
import matplotlib.image as img
import pandas as pd

# Problem 1

# Part a
n = 20
A1 = np.zeros((n, n), dtype=float)

for i in range(n):
    for j in range(n):
        A1[i, j] = 1.0 / (i + j + 1)

print("A1 = {}".format(A1))

# Part b

def frobenius_norm(A = A1):
    n, m = A.shape
    total = 0.0
    for i in range(n):
        for j in range(m):
            total += np.abs(A[i, j]) ** 2
    return  np.sqrt(total)

A2 = frobenius_norm(A1)
print("A2 = {}".format(A2))

# Part c
def infinity_norm(A = A1):
    n, m = A.shape
    rowMax = 0.0

    for i in range(n):
        rowsum = 0.0
        for j in range(m):
            rowsum += np.abs(A[i, j])
        if rowsum > rowMax:
            rowMax = rowsum
    return rowMax
A3 = infinity_norm(A1)
print("A3 = {}".format(A3))

# Part d
X = img.imread("hw6_img.png")
X = X[10:50, 10:50, 2]

A4 = frobenius_norm(X)
print("A4 = {}".format(A4))
A5 = infinity_norm(X)
print("A5 = {}".format(A5))

# Part e
def trace(A):
    n, m = A.shape
    trace = 0.0
    for i in range(n):
        for j in range(m):
            if i == j:
                trace += A[i, j]
    return trace
A6 = trace(A1)
print("A6 = {}".format(A6))
A7 = trace(X)
print("A7 = {}".format(A7))

# Problem 2
data = pd.read_csv("regression_data.csv")
x = data["x"].to_numpy()
y = data["y"].to_numpy()

# Part a

coeffs = np.polyfit(x, y, 1)
m = coeffs[0]
b = coeffs[1]
yhat = np.polyval(coeffs, x)

A8 = m
print("A8 = {}".format(A8))

# Part b
ymean = np.mean(y)
SSE = 0
for i in range(x.size):
    SSE += (y[i] - yhat[i]) ** 2
SST = 0
for i in range(x.size):
    SST += (y[i] - ymean) ** 2

A9 = 1 - SSE / SST
print("A9 = {}".format(A9))

# Part c
N = x.size
A10 = np.sqrt(np.sum((yhat - y) ** 2) / N)
print("A10 = {}".format(A10))

# Part d
A11 = m * 31.4 + b
print("A11 = {}".format(A11))

# Part e
coeffs1 = np.polyfit(x, y, 3)
a = coeffs1[0]
b = coeffs1[1]
c = coeffs1[2]
d = coeffs1[3]
A12 = b
print("A12 = {}".format(A12))

# Part f
yhat1 = np.polyval(coeffs1, x)
A13 = np.sqrt(np.sum((yhat1 - y) ** 2) / N)
print("A13 = {}".format(A13))

# Part g
coeffs3 = np.polyfit(y, x, 1)
m = coeffs3[0]
b = coeffs3[1]
A14 = m
print("A14 = {}".format(A14))

A15 = (31.4 - b) / m
print("A15 = {}".format(A15))

# Problem 3
im = img.imread("hw6_img.png")

# Part a
A16 = im[:, :, 0].copy()
# print("A16 = {}".format(A16))

# Part b
im_gs = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]) / 3.0
A17 = im_gs.copy()
# print("A17 = {}".format(A17))

# Part c
h, w = im_gs.shape
padded = np.zeros((h + 2, w + 2))
padded[1:-1, 1:-1] = im_gs
im_sobel_1 = np.zeros((h, w))

for i in range(h):
    for j in range(w):
        im_sobel_1[i, j] = (
            -1 * padded[i,     j    ] + 0 * padded[i,     j+1] +  1 * padded[i,     j+2] +
            -2 * padded[i+1,   j    ] + 0 * padded[i+1,   j+1] +  2 * padded[i+1,   j+2] +
            -1 * padded[i+2,   j    ] + 0 * padded[i+2,   j+1] +  1 * padded[i+2,   j+2]
        )

A18 = im_sobel_1.copy()
# print("A18= {}".format(A18))

# Part d
im_sobel_2 = np.zeros((h, w))
for i in range(h):
    for j in range(w):
        im_sobel_2[i, j] = (
            -1 * padded[i,     j    ] + -2 * padded[i,     j+1] + -1 * padded[i,     j+2] +
             0 * padded[i+1,   j    ] +  0 * padded[i+1,   j+1] +  0 * padded[i+1,   j+2] +
             1 * padded[i+2,   j    ] +  2 * padded[i+2,   j+1] +  1 * padded[i+2,   j+2]
        )

A19 = im_sobel_2
# print("A19 = {}".format(A19))

# Part e
A20 = np.sqrt(im_sobel_1 ** 2 + im_sobel_2 ** 2)
# print("A20 = {}".format(A20))
