import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

def fractional_derivative(f, alpha, x_val):
    n = int(np.ceil(alpha))
    coefficient = 1/gamma(n - alpha)

    def integrand(t):
        return (x_val - t)**(n - alpha - 1) * f(t)

    integral_value, _ = quad(integrand, -np.inf, x_val)
    return coefficient * integral_value

def original_function(x):
    return 0.1 * np.sin(x) if 0 <= x <= 4 else 0

def modified_function(x):
    if 0 <= x <= 1 or 3 <= x <= 4:
        return 0.1 * (np.sin(x) + 5 * np.sin(5 * x))
    elif 1 < x < 3:
        return 0.1 * np.sin(x)
    return 0

x_vals = np.linspace(0, 4, 400)
original_values = [original_function(x) for x in x_vals]
modified_values = [modified_function(x) for x in x_vals]

alpha = 0.5
y_orig_frac_derivative = [fractional_derivative(original_function, alpha, xi) for xi in x_vals]
y_mod_frac_derivative = [fractional_derivative(modified_function, alpha, xi) for xi in x_vals]

# x_vals = np.linspace(0, 4, 400)
# original_values = x_vals**2
# modified_values = np.where((x_vals > 3) | (x_vals < 1), x_vals**2 + 5 * np.sin(5 * x_vals), x_vals**2)

# alpha = 0.5
# y_orig_frac_derivative = [fractional_derivative(lambda x: np.sin(x), alpha, xi) for xi in x_vals]
# y_mod_frac_derivative = [fractional_derivative(lambda x: np.where(x > 3 or x < 1, np.sin(x) + 5 * np.sin(5 * x), np.sin(x)), alpha, xi) for xi in x_vals]

plt.figure(figsize=(12, 6))

# Plotting the original function and its fractional derivative
plt.subplot(1, 2, 1)
plt.plot(x_vals, original_values, label='f(x) = sin(x)')
plt.plot(x_vals, y_orig_frac_derivative, '--', label=f"f^{alpha}(x)")
plt.ylim(-1,1)
plt.axvline(2, color='red', linestyle='-.', label='x=2')
plt.legend()
plt.title("Original Function & Its Fractional Derivative")
plt.grid(True)

# Plotting the modified function and its fractional derivative
plt.subplot(1, 2, 2)
plt.plot(x_vals, modified_values, label='Modified f(x)')
plt.plot(x_vals, y_mod_frac_derivative, '--', label=f"Modified f^{alpha}(x)")
plt.ylim(-1, 1)
plt.axvline(2, color='red', linestyle='-.', label='x=2')
plt.legend()
plt.title("Modified Function & Its Fractional Derivative")
plt.grid(True)

plt.tight_layout()
plt.show()

index_at_x2 = np.argmin(np.abs(x_vals - 2))

print(f"Original function's fractional derivative at x=2: {y_orig_frac_derivative[index_at_x2]}")
print(f"Modified function's fractional derivative at x=2: {y_mod_frac_derivative[index_at_x2]}")
