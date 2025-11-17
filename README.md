# Scientific Computing – Numerical Methods Playground

This repository collects a set of small, focused Python scripts exploring core topics in numerical analysis and scientific computing: series approximations, root-finding, finite differences, ODE solvers, linear systems, regression, image processing, and Markov chains.  

Each file is self-contained and can be run independently.

---

## Repository Structure

> **Note:** Replace the placeholders like `series_collatz.py` with your actual filenames if they differ.

### 1. Series Approximations & Collatz Experiments  
**(e.g., `series_collatz.py`)**

- Approximates values related to the Riemann zeta function:
  - Partial sums of \(\sum 1/n^2\) and \(\sum 1/n^6\) to approximate \(\pi\).
  - Uses known constants (e.g., \(\pi^2/6\), \(\pi^6/945\)) and compares the numerical approximation to `np.pi`.
- Experiments with the **Collatz (3x + 1) problem**:
  - Iteration sequences for specific starting values (7, 15, 2025, etc.).
  - Searches for numbers with a specified stopping time.
  - Tracks the maximum value encountered in trajectories.
- Works with a slowly divergent series \(\sum 1/\log(n)\) to:
  - Compute partial sums.
  - Find how many terms are needed to exceed given thresholds.

**Concepts:** series convergence, numerical approximation of \(\pi\), integer dynamics, heuristic exploration of Collatz orbits.

---

### 2. Matrix Norms, Regression & Sobel Edge Detection  
**(e.g., `matrix_norms_regression_sobel.py`)**

- Builds a **Hilbert-like matrix** \(H_{ij} = 1/(i+j+1)\) and computes:
  - Frobenius norm (custom implementation).
  - Infinity norm (row-sum norm, custom implementation).
  - Trace of a matrix.
- Loads a small image patch (`hw6_img.png`) and:
  - Computes Frobenius and infinity norms of the patch.
  - Computes the trace for both the Hilbert matrix and the image slice.
- Performs **least-squares regression** from CSV data:
  - Fits linear and cubic polynomials with `np.polyfit`.
  - Computes \(R^2\) by hand from SSE and SST.
  - Evaluates prediction at a specific point (e.g., `x = 31.4`).
  - Compares RMSE between linear and cubic fits.
- Uses **Sobel filters** for simple edge detection:
  - Converts RGB to grayscale.
  - Applies horizontal and vertical Sobel kernels via explicit loops.
  - Combines gradients to get gradient magnitude.

**Concepts:** norms, regression, model fit quality, basic image processing, convolution, edge detection.

---

### 3. Finite Difference BVPs, ODE Time-Stepping & Markov Weather Model  
**(e.g., `bvp_odes_markov_weather.py`)**

#### Boundary Value Problem (BVP)
- Discretizes a second-order ODE with first-derivative term using **finite differences** on \([0,\pi]\) with Dirichlet boundary conditions.
- Constructs a tridiagonal system \(L u = b\):
  - Derives coefficients from the discretization step \(h = \pi/N\).
  - Fills the full \((N+1)\times(N+1)\) matrix explicitly.
- Solves the linear system with `scipy.linalg.solve`.
- Compares the numerical solution to a known **exact solution** \(u_{\text{exact}}(x)\) and computes the max error.

#### Linear ODE System – Time-Stepping Methods
System:
\[
\begin{cases}
x' = -x + 3y + 1 \\
y' = -x - 2y + 2
\end{cases}
\]
- Implements multiple time integrators over \(t \in [0,1]\) with step `dt = 0.01`:
  - Forward Euler.
  - Backward Euler (solving a 2×2 system each step).
  - Explicit trapezoid (improved/Heun’s method).
  - Implicit trapezoid (Crank–Nicolson form).
- Compares the final states from each method.

#### Markov Chain Weather Model
- Defines a 3-state transition matrix \(P\) (e.g., [sunny, cloudy, rainy]).
- Evolves different initial distributions:
  - 1-step transition from “rainy”.
  - New Year’s Eve distribution → New Year’s Day.
  - 365-step evolution from different starting states.
- Compares the long-term distributions from different initial conditions and computes their absolute difference.

**Concepts:** finite difference discretization, linear systems for BVPs, explicit vs implicit ODE solvers, Markov chains, stationary behavior.

---

### 4. Nonlinear Root Finding: Bisection, False Position, Newton & Fixed-Point Iteration  
**(e.g., `root_finding_methods.py`)**

- Uses a cubic test function:
  \[
  f(x) = x^3 - \frac{6}{5}x^2 - \frac{9}{5}x + \frac{1}{2}
  \]
  and evaluates it at test points.
- Implements **bisection**:
  - Custom function returning the root estimate, number of iterations, and final interval length.
  - Used on different intervals to isolate roots.
- Implements **false position (regula falsi)**:
  - Linear interpolation between endpoints to refine the root.
- Implements **Newton’s method**:
  - Tested on \(g(x) = x\), \(h(x) = x^2\), and \(i(x) = x^{51}\) to see convergence (and potential issues).
- Compares **Newton vs bisection** in terms of iteration counts.
- Implements a **fixed-point iteration**:
  - Example: \(g(x) = \sin(x) + 0.5\).
  - Iterative method with stopping criteria based on \(|x - g(x)|\).
  - Compares convergence from very different starting points (e.g., 0 vs 2025).

**Concepts:** bracketing vs open methods, convergence behavior, sensitivity to starting guess, fixed-point iteration.

---

### 5. Finite Difference Derivatives, Population Data & Secant Method for √2  
**(e.g., `finite_diff_population_secant.py`)**

#### Finite Difference Derivative Approximations
- At \(x_0 = \pi/4\), with \(f(x) = x\sin x\):
  - Computes analytic first and second derivatives.
  - Implements forward, backward, central, and higher-order finite difference schemes.
  - Compares approximations at two step sizes (`dx = 0.01` and `0.001`).
  - Computes the **observed order of accuracy** via error ratios \(q\).

#### Population Data – Derivatives & Integration
- Loads `population.csv` into `pandas`.
- Constructs:
  - Discrete derivative \(N'(t)\) using forward, central, and backward differences at the endpoints.
  - Left-hand rule, right-hand rule, and trapezoid rule to approximate the integral of \(N(t)\) over time.
- Extracts specific derivative values and full derivative arrays for inspection.

#### Secant Method for \(\sqrt{2}\)
- Defines \(h(x) = x^2 - 2\).
- Implements several variants of the **secant method**:
  - One that stores the sequence of guesses and absolute errors across iterations.
  - Variants that:
    - Stop based on error in \(x\) versus \(\sqrt{2}\).
    - Stop based on \(|f(x)|\).
    - Stop based on \(|x_{n} - x_{n-1}|\).
- Tests convergence starting from far-out initial guesses (e.g., 2025, 2024).

**Concepts:** finite difference differentiation, numerical integration rules, error analysis, convergence criteria for secant method.

---

### 6. Euler Time-Stepping for ODEs and Systems: Forward/Backward Schemes & Error Analysis  
**(e.g., `euler_odes_systems.py`)**

#### Single ODE – Damped Oscillator-type Equation
- ODE with known exact solution:
  \[
  x'(t) = -0.1x - \Bigl(1 + \frac{0.1^2}{4}\Bigr)e^{-0.1 t/2}\sin t, \quad
  x_{\text{exact}}(t) = e^{-0.1 t/2}\Bigl(\cos t - \frac{0.1}{2}\sin t\Bigr)
  \]
- Implements:
  - Forward Euler.
  - Backward Euler with closed-form step update.
- Compares approximate solutions at \(T=20\) for \(dt=0.1\) and \(0.01\), and computes absolute error vs exact solution.

#### Nonlinear ODE – Backward Euler via Bisection
- ODE:
  \[
  y'(t) = 8\sin(y)
  \]
- Forward Euler:
  - Integrates from \(t=0\) to \(T=2\) with different `dt`.
  - Computes max error vs exact solution \(y_{\text{exact}}(t)\).
- Backward Euler:
  - Implicit step solved with a nested **bisection** method at each time step.
  - Compares error vs forward Euler.

#### 2D Linear System
\[
\begin{cases}
x' = 2x - 2y \\
y' = x
\end{cases}, \quad (x(0), y(0)) = (1,-1)
\]
- Forward Euler system solver `feSys`.
- Backward Euler system solver `be_sys` using an analytic update formula.
- Compares values at \(t = 4\) and computes the difference between methods.

**Concepts:** local vs global error, convergence rates, implicit vs explicit schemes, solving implicit steps via root finding, systems of ODEs.

---

## Requirements

- Python 3.x
- `numpy`
- `scipy` (for linear solves in the BVP/ODE file)
- `pandas` (for CSV-based data processing)
- `matplotlib` (for image I/O and potential plotting, e.g., `matplotlib.image`)

Install dependencies (example):

```bash
pip install numpy scipy pandas matplotlib
