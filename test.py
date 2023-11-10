import matplotlib.pyplot as plt

def mackey_glass(x, beta, gamma, tao, n):
    # if len(x) >= tao + 1:
    return x + (beta * x[-tao-1] / (1 + x[-tao-1]**n)) - (gamma * x[-1])
    # else:
    #     return x[-1]  

# Initial value
x = 0.1

# Number of iterations
num_iterations = 100

# Store all the values
values = [x]

# Iterate and generate the Tent Map series
for _ in range(num_iterations):
    x = mackey_glass(x, beta = 0.2, gamma = 0.1, tao = 17, n=10)
    values.append(x)

# Plot the values
plt.figure(figsize=(10, 6))
plt.plot(values)
plt.title('Tent Map')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.grid(True)
plt.show()
