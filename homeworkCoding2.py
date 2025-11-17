import numpy as np

# Problem 1

# Part A
values = np.zeros(10)
sumA = 0

for x in range(values.size):
    values[x] = 1/ ((x + 1) ** 2)
    sumA += values[x]

A1 = sumA
print(sumA)

# Part B
sumB = 0
M = 0

while sumB <= 1.644934:
    currentValue = 1 / ((M + 1) ** 2)
    sumB += currentValue
    M += 1

A2 = M - 1
A3 = sumB
print(A2)
print(A3)

# Part C
A4 = (6 * A3) ** (1/2)
A5 = np.abs(A4 - np.pi)
print(A4)
print(A5)

#Part D
sumD= 0

for x in range(A2):
    currentValue = 1 / ((x + 1) ** 6)
    sumD += currentValue

A6 = (sumD * 945) ** (1/6)
A7 = np.abs(A6 - np.pi)
print(A6)
print(A7)

# Problem 2

# Part A
values = np.zeros(4)
counter = 0
x = 7

while counter <= 3:
    if x % 2 == 0:
        x =  x / 2
        values[counter] = x
        counter += 1
    elif x % 2 == 1:
        x = x * 3 + 1
        values[counter] = x
        counter += 1

A8 = values[:]
print(A8)

# Part B
x = 15
counter = 0

while x != 1:
    if x % 2 == 0:
        x = x / 2
        counter += 1
    elif x % 2 == 1:
        x = x * 3 + 1
        counter += 1

A9 = counter
print(A9)

# Part C
x = 2025
counter = 0

while x != 1:
    if x % 2 == 0:
        x = x / 2
        counter += 1
    elif x % 2 == 1:
        x = x * 3 + 1
        counter += 1

A10 = counter
print(A10)

# Part D
z = 0

for x in range(1, 30):
    y = x
    counter = 0

    while counter <= 10 and y != 1:
        if y % 2 == 0:
            y /= 2
        elif y % 2 == 1:
            y = 3 * y + 1
        counter += 1

    if y == 1 and counter == 10:
        z = x
        break

A11 = z
print(A11)

# Part E
values = np.zeros(10000)
counter = 0
greater = 0
x = 1000

while counter <= 9999:
    if x % 2 == 0:
        x =  x / 2
        values[counter] = x
    elif x % 2 == 1:
        x = x * 3 + 1
        values[counter] = x
    counter += 1

A12 = np.max(values)
print(A12)

# Part F
steps = np.zeros(100)

for x in range(1, 100):
    y = x
    counter = 0

    while y != 1:
        if y % 2 == 0:
            y /= 2
        elif y % 2 == 1:
            y = 3 * y + 1
        counter += 1
    steps[x] = counter

A13 = int(np.argmax(steps[1:])) + 1  # map slice index (0..99) back to 1..100                            # p with the largest stopping time
A14 = int(steps[A13])
print(A13)
print(A14)


# Problem 3

# Part A
values = np.zeros(4)
sumA = 0
N = 2

for x in range(values.size):
    values[x] = 1 / np.log(N)
    sumA += values[x]
    N += 1

A15 = sumA
print(A15)

# Part B

values = np.zeros(2024)
sumB = 0
N = 2

for x in range(values.size):
    values[x] = 1 / np.log(N)
    sumB += values[x]
    N += 1

A16 = sumB
print(A16)

# Part C
sumC = 0
N = 2

while sumC < 10:
    currentValue = 1 / np.log(N)
    sumC += currentValue
    N += 1

A17 = N - 1
A18 = sumC
print(A17)
print(A18)


# Part D
sumD = 0
N = 2

while sumD <= 1_000_000:
    currentValue = 1 / np.log(N)
    sumD += currentValue
    N += 1

A19 = N - 1
A20 = sumD
print(A19)
print(A20)
