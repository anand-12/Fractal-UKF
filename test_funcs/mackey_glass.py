def mackey_glass(x, beta, gamma, tao, n):
    if len(x) >= tao + 1:
        return x[-1] + (beta * x[-tao-1] / (1 + x[-tao-1]**n)) - (gamma * x[-1])
    else:
        return x[-1]  # Return the last known value when there's not enough history