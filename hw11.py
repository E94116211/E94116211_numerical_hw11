import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

h = 0.1
x_vals = np.arange(0, 1 + h, h)

# ---- (a) Shooting Method ---- #
def odes(x, y):
    dy1 = y[1]
    dy2 = -(x + 1) * y[1] + 2 * y[0] + (1 - x**2) * np.exp(-x)
    return [dy1, dy2]

def shooting_function(guess):
    sol = solve_ivp(odes, [0, 1], [1, guess], t_eval=[1])
    return sol.y[0, -1] - 2

sol_guess = root_scalar(shooting_function, bracket=[0, 5], method='brentq')
correct_slope = sol_guess.root
sol_shoot = solve_ivp(odes, [0, 1], [1, correct_slope], t_eval=x_vals)
y_shoot = sol_shoot.y[0]

# ---- (b) Finite Difference Method ---- #
n = int(1 / h)
x_fd = np.linspace(0, 1, n + 1)
A = np.zeros((n - 1, n - 1))
b = np.zeros(n - 1)

for i in range(1, n):
    xi = x_fd[i]
    a = 1/h**2 + (xi + 1)/(2*h)
    c = 1/h**2 - (xi + 1)/(2*h)
    d = -2/h**2 + 2

    if i > 1:
        A[i - 1, i - 2] = a
    A[i - 1, i - 1] = d
    if i < n - 1:
        A[i - 1, i] = c

    rhs = (1 - xi**2) * np.exp(-xi)
    if i == 1:
        rhs -= a * 1
    if i == n - 1:
        rhs -= c * 2

    b[i - 1] = rhs

y_fd_internal = np.linalg.solve(A, b)
y_fd = np.concatenate(([1], y_fd_internal, [2]))

# ---- (c) Variational Method ---- #
def basis_function(x, i):
    return x * (1 - x) * x**i

m = 5  # Number of basis functions
x_var = x_vals
phi = np.array([[basis_function(xi, i) for i in range(m)] for xi in x_var])
y0 = 1 + x_var  # Satisfies boundary conditions

K = np.zeros((m, m))
F = np.zeros(m)

for i in range(m):
    for j in range(m):
        integrand = (-(x_var + 1) * np.gradient(phi[:, j], h) +
                     2 * phi[:, j]) * phi[:, i]
        K[i, j] = np.sum(integrand) * h

    rhs = (-(x_var + 1) * np.gradient(y0, h) + 2 * y0 + (1 - x_var**2) * np.exp(-x_var))
    F[i] = -np.sum(rhs * phi[:, i]) * h

c = np.linalg.solve(K, F)
y_var = y0 + phi @ c

# ---- Plot All Results ---- #
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_shoot, 'o-', label='(a) Shooting Method')
plt.plot(x_fd, y_fd, 's--', label='(b) Finite Difference')
plt.plot(x_var, y_var, 'd-.', label='(c) Variational Method')
plt.title("Comparison of Numerical Methods for BVP")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()