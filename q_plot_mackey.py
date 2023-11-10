import matplotlib.pyplot as plt
import numpy as np


# Provided H values for each case
H_values_mackey_30 = [0.3727,0.3714,0.3702,0.3689,0.3677,0.3664,0.3652,0.3639,0.3626,0.3614,0.3601,0.3589,0.3576,0.3563,0.3551,0.3538,0.3525,0.3513,0.3500,0.3487,0.3474,0.3462,0.3449,0.3436,0.3424,0.3411,0.3398,0.3386,0.3373,0.3360,0.3348,0.3335,0.3323,0.3310,0.3297,0.3285,0.3272,0.3260,0.3247,0.3235,0.3222,0.3210,0.3197,0.3185,0.3172,0.3160,0.3147,0.3135,0.3123,0.3110,0.3098,0.3086,0.3073,0.3061,0.3049,0.3036,0.3024,0.3012,0.3000,0.2987,0.2975,0.2963,0.2951,0.2939,0.2927,0.2915,0.2903,0.2891,0.2879,0.2866,0.2856,0.2845,0.2835,0.2824,0.2814,0.2804,0.2794,0.2783,0.2773,0.2763,0.2753,0.2742,0.2732,0.2722,0.2712,0.2701,0.2691,0.2681,0.2671,0.2661,0.2651,0.2642,0.2632,0.2623,0.2613,0.2604,0.2595,0.2585,0.2576,0.2567] # Fill in with all the H values for tau = 30
H_values_mackey_22 = [0.3300, 0.3287, 0.3273, 0.3260, 0.3247, 0.3233, 0.3220, 0.3207, 0.3193, 0.3180, 0.3166, 0.3153, 0.3140, 0.3126, 0.3113, 0.3100, 0.3086, 0.3074, 0.3062, 0.3050, 0.3038, 0.3026, 0.3015, 0.3003, 0.2991, 0.2979, 0.2967, 0.2955, 0.2943, 0.2931, 0.2919, 0.2907, 0.2895, 0.2883, 0.2871, 0.2859, 0.2847, 0.2835, 0.2823, 0.2811, 0.2800, 0.2788, 0.2777, 0.2765, 0.2754, 0.2742, 0.2731, 0.2719, 0.2708, 0.2696, 0.2685, 0.2674, 0.2662, 0.2651, 0.2639, 0.2628, 0.2617, 0.2605, 0.2594, 0.2583, 0.2571, 0.2560, 0.2548, 0.2538, 0.2528, 0.2518, 0.2508, 0.2498, 0.2488, 0.2478, 0.2468, 0.2458, 0.2449, 0.2439, 0.2429, 0.2419, 0.2409, 0.2399, 0.2389, 0.2379, 0.2369, 0.2360, 0.2350, 0.2340, 0.2330, 0.2320, 0.2311, 0.2316, 0.2325, 0.2334, 0.2342, 0.2351, 0.2360, 0.2368, 0.2377, 0.2386, 0.2394, 0.2403, 0.2411, 0.2420] # Fill in with all the H values for tau = 22
H_values_mackey_17 = [0.2104, 0.2091, 0.2078, 0.2065, 0.2051, 0.2038, 0.2025, 0.2012, 0.1999, 0.1986, 0.1973, 0.1959, 0.1946, 0.1933, 0.1920, 0.1907, 0.1894, 0.1880, 0.1867, 0.1854, 0.1841, 0.1828, 0.1815, 0.1802, 0.1789, 0.1776, 0.1763, 0.1750, 0.1737, 0.1725, 0.1712, 0.1699, 0.1686, 0.1673, 0.1661, 0.1648, 0.1636, 0.1625, 0.1614, 0.1602, 0.1601, 0.1611, 0.1620, 0.1630, 0.1640, 0.1649, 0.1658, 0.1668, 0.1677, 0.1686, 0.1696, 0.1705, 0.1714, 0.1723, 0.1732, 0.1741, 0.1750, 0.1759, 0.1768, 0.1777, 0.1786, 0.1794, 0.1803, 0.1812, 0.1821, 0.1829, 0.1838, 0.1846, 0.1855, 0.1863, 0.1872, 0.1880, 0.1889, 0.1897, 0.1906, 0.1914, 0.1922, 0.1931, 0.1939, 0.1947, 0.1955, 0.1964, 0.1972, 0.1980, 0.1988, 0.1996, 0.2004, 0.2012, 0.2020, 0.2028, 0.2036, 0.2044, 0.2052, 0.2060, 0.2069, 0.2077, 0.2085, 0.2093, 0.2101, 0.2109] # Fill in with all the H values for tau = 17
plt.style.use('seaborn-darkgrid')

# Create a color palette
palette = plt.get_cmap('tab10')

# q values ranging from 1.1 to 2.1 in steps of 0.01
q_values = np.arange(1.1, 2.1, 0.01)


points_to_mark = {
    "tau = 30": [
        # (None, 0.2659, "True (30)"),
        # (None, 0.4872, "UKF (30)"),
        (1.9900000000000009, 0.2660880898870356, "Optimal_Mackey_30")
    ],
    "tau = 22": [
        # (None, 0.1832, "True (22)"),
        # (None, 0.4467, "UKF (22)"),
        (1.9600000000000009, 0.2311101372511777, "Optimal_Mackey_22")
    ],
    "tau = 17": [
        # (None, 0.1036, "True (17)"),
        # (None, 0.3380, "UKF (17)"),
        (1.5000000000000004, 0.16010560316294836, "Optimal_Mackey_17")
    ],
}

# Plot
plt.figure(figsize=(12, 8))
plt.plot(q_values, H_values_mackey_30, label='Mackey:tau = 30')
plt.plot(q_values, H_values_mackey_22, label='Mackey:tau = 22')
plt.plot(q_values, H_values_mackey_17, label='Mackey:tau = 17')


# Adding markers and annotations
for tau, points in points_to_mark.items():
    for q, h, label in points:
        if "Optimal" not in label:
            plt.axhline(h, color='green', linestyle='-', linewidth=1)  # Horizontal line across the axis
            plt.text(min(q_values)-0.05, h, label, fontsize=9, verticalalignment='center', horizontalalignment='right')
        else:
            plt.scatter(q, h, s=100, marker='o')
            plt.text(q, h, label, fontsize=9, verticalalignment='bottom')

# Other chart settings
plt.xlabel('q values', fontsize=14)
plt.ylabel('Hurst Exponent (H)', fontsize=14)
plt.title('Hurst Exponent (H) vs. q values', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Adding minor grid for better precision
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

plt.tight_layout()
plt.show()


