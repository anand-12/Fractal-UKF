import numpy as np
import matplotlib.pyplot as plt

def tent_map(x, mu=1.99):
    return mu * np.where(x < 0.5, x, 1-x)

def bifurcation_tent_map(mu_values=np.linspace(1, 1.99, 1000), iterations=1000, last=100, x0=0.5):
    x = np.ones(len(mu_values)) * x0
    for i in range(iterations):
        x = tent_map(x, mu_values)
        if i >= (iterations - last):
            plt.plot(mu_values, x, ',r', alpha=0.8)
    plt.xlabel('µ')
    plt.ylabel('$x_n$')
    plt.title('Bifurcation Diagram of Tent Map')
    plt.show()

def logistic_map(x, mu=4):
    return mu * x * (1 - x)

def bifurcation_logistic_map(mu_values=np.linspace(2.5, 4, 1000), iterations=1000, last=100, x0=0.5):
    x = np.ones(len(mu_values)) * x0
    for i in range(iterations):
        x = logistic_map(x, mu_values)
        if i >= (iterations - last):
            plt.plot(mu_values, x, ',k', alpha=0.25)
    plt.xlabel('µ')
    plt.ylabel('$x_n$')
    plt.title('Bifurcation Diagram of Logistic Map')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def mackey_glass_step(x_t, x_t_minus_tau, beta=0.2, gamma=0.1, n=10):
    return beta * x_t_minus_tau / (1 + x_t_minus_tau**n) - gamma * x_t

def mackey_glass_trajectory(tau, beta=0.2, gamma=0.1, n=10, iterations=10000, dt=0.1, x0=0.9):
    x = np.zeros(iterations)
    x[0:int(tau/dt)] = x0
    for i in range(int(tau/dt), iterations-1):
        x[i+1] = x[i] + dt * mackey_glass_step(x[i], x[i-int(tau/dt)], beta, gamma, n)
    return x

def plot_mackey_glass_attractor(tau_values=[25, 22, 20, 17, 15, 13, 11], discard_iterations=2000):
    for tau in tau_values:
        x = mackey_glass_trajectory(tau)
        x = x[discard_iterations:]  # Discard the first 'discard_iterations' data points
        plt.figure()
        plt.plot(x[:-int(tau/0.1)], x[int(tau/0.1):], '-', markersize=0.25)
        plt.title(f'Mackey-Glass Attractor for τ = {tau}')
        plt.xlabel('$x(t)$')
        plt.ylabel(f'$x(t-{tau})$')
        plt.grid(True)
    plt.show()

plot_mackey_glass_attractor()



# def mackey_glass_step(x_t, x_t_minus_tau, beta=0.2, gamma=0.1, n=10):
#     return beta * x_t_minus_tau / (1 + x_t_minus_tau**n) - gamma * x_t

# def find_local_maxima(ts_data):
#     return [i for i in range(1, len(ts_data)-1) if ts_data[i] > ts_data[i-1] and ts_data[i] > ts_data[i+1]]

# def mackey_glass_bifurcation_3d(beta_values=np.linspace(0.1, 2, 20), tau_values=np.linspace(15, 25, 20), gamma=0.1, n=10, iterations=5000, dt=0.1, x0=0.5):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     for beta in beta_values:
#         for tau in tau_values:
#             x = np.zeros(iterations)
#             x[0:int(tau/dt)] = x0
#             for i in range(int(tau/dt), iterations-1):
#                 x[i+1] = x[i] + dt * mackey_glass_step(x[i], x[i-int(tau/dt)], beta, gamma, n)
            
#             local_max_indices = find_local_maxima(x[-int(iterations*0.3):])
#             max_values = [x[i] for i in local_max_indices]
#             ax.scatter([beta]*len(local_max_indices), [tau]*len(local_max_indices), max_values, c='k', marker='o')

#     ax.set_xlabel('Beta (β)')
#     ax.set_ylabel('Tau (τ)')
#     ax.set_zlabel('Local Maxima of $x(t)$')
#     ax.set_title('3D Bifurcation Diagram for Mackey-Glass Equation')
#     plt.show()

# mackey_glass_bifurcation_3d()

# bifurcation_logistic_map()

# bifurcation_tent_map()    
