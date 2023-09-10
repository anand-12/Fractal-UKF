import numpy as np
from matplotlib import pyplot as plt
# def generate_fBm(length, hurst_exponent):
#     """
#     Generate a fractional Brownian motion (fBm) time series with a specified Hurst exponent.

#     Args:
#         length (int): Length of the fBm series.
#         hurst_exponent (float): Hurst exponent controlling the self-similarity.
        
#     Returns:
#         numpy.ndarray: An fBm time series.
#     """
#     # Create a vector of random Gaussian increments
#     increments = np.random.randn(length)
    
#     # Initialize the fBm series with zeros
#     fbm_series = np.zeros(length)
    
#     # Calculate the cumulative sum with scaling based on the Hurst exponent
#     for i in range(1, length):
#         fbm_series[i] = fbm_series[i-1] + (increments[i] / (2.0**hurst_exponent))
    
#     return fbm_series


# data = generate_fBm(1024, 0.7)
# # print(fbm_series, type(fbm_series))

# x = np.arange(len(data))

# # Create the plot
# plt.plot(x, data)

# # Add labels and a title
# plt.xlabel('X-Axis Label')
# plt.ylabel('Y-Axis Label')
# plt.title('Numpy Series Plot')

# # Display the plot
# plt.show()

def generate_fBm(x, hurst_exponent):
    """
    Generate the next value in a fractional Brownian motion (fBm) time series.

    Args:
        x (list or array): History of previous values.
        hurst_exponent (float): Hurst exponent controlling the self-similarity.

    Returns:
        float: Next value in the fBm series.
    """
    if len(x) >= 1:
        # Calculate the next value using fBm model
        increment = np.random.randn()
        next_value = x[-1] + (increment / (2.0 ** hurst_exponent))
        return next_value
    else:
        # If there's not enough history, return the initial value
        return x[0]  # Return the initial value