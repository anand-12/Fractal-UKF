import numpy as np
import matplotlib.pyplot as plt

def original_function(x):
    return np.sin(x)

def modified_function(x):
    y = np.copy(np.sin(x))
    mask = (x > 3) | (x < 1)  # Modify the function outside of a neighborhood of x=2
    y[mask] += 5 * np.sin(5 * x[mask])
    return y


x_vals = np.linspace(0, 4, 400)

# Compute the derivatives using numpy's gradient function for simplicity
y_orig_vals = original_function(x_vals)
y_orig_derivative = np.gradient(y_orig_vals, x_vals)

y_mod_vals = modified_function(x_vals)
y_mod_derivative = np.gradient(y_mod_vals, x_vals)

plt.figure(figsize=(12, 6))

# Plotting the original function and its derivative
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_orig_vals, label='f(x) = sin(x)')
plt.plot(x_vals, y_orig_derivative, '--', label="f'(x)")
plt.ylim(-1, 1)
plt.legend()
plt.title("Original Function & Its Derivative")
plt.grid(True)

# Plotting the modified function and its derivative
plt.subplot(1, 2, 2)
plt.plot(x_vals, y_mod_vals, label='Modified f(x)')
plt.plot(x_vals, y_mod_derivative, '--', label="Modified f'(x)")
plt.ylim(-1,1)
plt.legend()
plt.title("Modified Function & Its Derivative")
plt.grid(True)

plt.tight_layout()
plt.show()

index_at_x2 = np.argmin(np.abs(x_vals - 2))

print(f"Original function's fractional derivative at x=2: {y_orig_derivative[index_at_x2]}")
print(f"Modified function's fractional derivative at x=2: {y_mod_derivative[index_at_x2]}")
