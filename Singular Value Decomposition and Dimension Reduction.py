import numpy as np
import scipy as scp
import pandas as pd


# Problem 1

# Part a
A = np.array([[ 3, -2,  2],
              [ 4,  0,  1],
              [-1,  2,  2],
              [ 1,  0,  3]], dtype=float)

U, s, Vt = scp.linalg.svd(A, full_matrices=False)
A1 = np.sort(s)[::-1]
print("A1 = {}".format(A1))

# Part b
UT, s_tilde, VtT = scp.linalg.svd(A.T, full_matrices=False)
s_tilde_sorted = np.sort(s_tilde)[::-1]

A2 = np.vstack((A1,
                s_tilde_sorted,
                np.abs(A1 - s_tilde_sorted)))
print("A2 = {}".format(A2))

# Part c
A3 = A.T @ A
print("A3 = {}".format(A3))

# Part d
U_B, sB, Vt_B = scp.linalg.svd(A3, full_matrices=False)
sigma = np.sort(sB)[::-1]

e_vals, e_vecs = scp.linalg.eig(A3)
lambda_vals = np.sort(e_vals.real)[::-1]

A4 = np.vstack((sigma,
                lambda_vals,
                np.abs(sigma - lambda_vals)))
print("A4 = {}".format(A4))

# Part e
A5 = np.max(np.abs(A1**2 - lambda_vals))
print("A5 = {}".format(A5))

# Part f
total_energy = np.sum(A1**2)
fro_sq = np.linalg.norm(A, 'fro')**2
A6 = np.array([total_energy, fro_sq, np.abs(total_energy - fro_sq)])
print("A6 = {}".format(A6))

# Part g
ratio = A1[0] / A1[2]
condA = np.linalg.cond(A)
A7 = np.array([ratio, condA, np.abs(ratio - condA)])
print("A7 = {}".format(A7))

# Problem 2

# Part a
n = 100
main = 2 * np.ones(n)
off = -1 * np.ones(n - 1)
A100 = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

e_vals, e_vecs = scp.linalg.eig(A100)
idx = np.argmax(np.abs(e_vals))
A8 = e_vals[idx].real
print("A8 = {}".format(A8))

# Part b
v = np.ones((n, 1))
v = v / np.linalg.norm(v)

lam = float((v.T @ (A100 @ v))[0, 0])
eps = 1e-5

while np.linalg.norm(A100 @ v - lam * v) >= eps:
    w = A100 @ v
    v = w / np.linalg.norm(w)
    lam = float((v.T @ (A100 @ v))[0, 0])

A9 = lam
A10 = v

print("A9 = {}".format(A9))
print("A10 shape =", A10.shape)

# Part c
n = 100
main = 2 * np.ones(n)
off = -1 * np.ones(n - 1)
A100 = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

v = np.ones((n, 1))
lam = float((v.T @ (A100 @ v))[0, 0])
eps = 1e-5

while np.linalg.norm(A100 @ v - lam * v) >= eps:
    w = scp.linalg.solve(A100, v)
    v = w / np.linalg.norm(w)
    lam = float((v.T @ (A100 @ v))[0, 0])

A11 = lam.real
A12 = v

print("A11 = {}".format(A11))
print("A12 shape =", A12.shape)


# Problem 3
df = pd.read_csv("coding_hw8_data.csv", header=None)
M = df.to_numpy()

# Part a
U, s, Vt = scp.linalg.svd(M, full_matrices=False)

M1 = s[0] * np.outer(U[:, 0], Vt[0, :])
A13 = M1

num = np.linalg.norm(M - M1, 'fro')
den = np.linalg.norm(M, 'fro')
e1 = num / den
A14 = e1

print("A13 shape =", A13.shape)
print("A14 =", A14)

# Part b
U, s, Vt = scp.linalg.svd(M, full_matrices=False)

idx = np.where(s < 1e-5)[0][0]
A15 = int(idx + 1)

k = A15
Uk = U[:, :k]
sk = s[:k]
Vtk = Vt[:k, :]

Mk = (Uk * sk) @ Vtk

num = np.linalg.norm(M - Mk, 'fro')
den = np.linalg.norm(M, 'fro')
A16 = num / den

print("A15 = {}".format(A15))
print("A16 = {}".format(A16))

# Part c
U, s, Vt = scp.linalg.svd(M, full_matrices=False)

E = np.cumsum(s**2) / np.sum(s**2)
e_all = np.sqrt(1 - E)

idx = np.where(e_all < 0.01)[0][0]
A17 = int(idx + 1)
A18 = e_all[idx]

print("A17 = {}".format(A17))
print("A18 = {}".format(A18))

# Part d
U, s, Vt = scp.linalg.svd(M, full_matrices=False)

A19 = np.sum(s**2)

k = A17
A20 = np.sum(s[:k]**2) / A19

print("A19 = {}".format(A19))
print("A20 = {}".format(A20))
