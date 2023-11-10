import matplotlib.pyplot as plt
import numpy as np


# Provided H values for each case
H_values_logmap_37 = [0.3952, 0.3831, 0.3720, 0.3615, 
0.3520, 0.3437, 0.3361, 0.3291, 
0.3226, 0.3164, 0.3115, 0.3070, 
0.3025, 0.2982, 0.2942, 0.2921, 
0.2903, 0.2885, 0.2869, 0.2853, 
0.2838, 0.2819, 0.2799, 0.2779, 
0.2759, 0.2719, 0.2678, 0.2637, 
0.2595, 0.2554, 0.2512, 0.2463, 
0.2411, 0.2349, 0.2287, 0.2223, 
0.2159, 0.2094, 0.2180, 0.2274, 
0.2361, 0.2418, 0.2473, 0.2528, 
0.2581, 0.2634, 0.2684, 0.2734, 
0.2783, 0.2827, 0.2868, 0.2910, 
0.2939, 0.2967, 0.2992, 0.3017, 
0.3042, 0.3068, 0.3094, 0.3120, 
0.3145, 0.3170, 0.3200, 0.3229, 
0.3259, 0.3290, 0.3321, 0.3352, 
0.3384, 0.3417, 0.3450, 0.3482, 
0.3515, 0.3549, 0.3583, 0.3617, 
0.3652, 0.3687, 0.3722, 0.3758, 
0.3794, 0.3830, 0.3867, 0.3903, 
0.3940, 0.3977, 0.4013, 0.4046, 
0.4078, 0.4110, 0.4139, 0.4169, 
0.4198, 0.4227, 0.4258, 0.4286, 
0.4314, 0.4342]
H_values_logmap_38 = [0.1896, 
0.1891, 
0.1893, 
0.1898, 
0.1903, 
0.1904, 
0.1901, 
0.1891, 
0.1880, 
0.1862, 
0.1842, 
0.1826, 
0.1813, 
0.1799, 
0.1786, 
0.1772, 
0.1757, 
0.1740, 
0.1723, 
0.1704, 
0.1684, 
0.1663, 
0.1642, 
0.1618, 
0.1594, 
0.1569, 
0.1544, 
0.1519, 
0.1493, 
0.1467, 
0.1440, 
0.1396, 
0.1344, 
0.1289, 
0.1250, 
0.1247, 
0.1244, 
0.1244, 
0.1343, 
0.1425, 
0.1505, 
0.1568, 
0.1622, 
0.1675, 
0.1727, 
0.1779, 
0.1830, 
0.1880, 
0.1928, 
0.1975, 
0.2020, 
0.2065, 
0.2109, 
0.2152, 
0.2194, 
0.2229, 
0.2258, 
0.2285, 
0.2307, 
0.2327, 
0.2345, 
0.2356, 
0.2366, 
0.2376, 
0.2384, 
0.2387, 
0.2389, 
0.2391, 
0.2395, 
0.2414, 
0.2432, 
0.2451, 
0.2464, 
0.2471, 
0.2479, 
0.2485, 
0.2490, 
0.2496, 
0.2518, 
0.2557, 
0.2593, 
0.2628, 
0.2664, 
0.2702, 
0.2743, 
0.2784, 
0.2825, 
0.2867, 
0.2910, 
0.2952, 
0.2996, 
0.3039, 
0.3083, 
0.3127, 
0.3171, 
0.3216, 
0.3260, 
0.3305]
H_values_logmap_399 = [0.2949, 
0.2813, 
0.2680, 
0.2765, 
0.2843, 
0.2915, 
0.2981, 
0.2992, 
0.2994, 
0.2994, 
0.2994, 
0.2993, 
0.2991, 
0.2988, 
0.2985, 
0.2980, 
0.2974, 
0.2968, 
0.2961, 
0.2955, 
0.2949, 
0.2942, 
0.2936, 
0.2930, 
0.2924, 
0.2918, 
0.2911, 
0.2904, 
0.2897, 
0.2890, 
0.2883, 
0.2876, 
0.2868, 
0.2860, 
0.2853, 
0.2846, 
0.2866, 
0.2890, 
0.2923, 
0.2962]
# Using a professional plot style
q_values = np.arange(0.01, 1.5, 0.01)
print(len(H_values_logmap_399), len(H_values_logmap_38), len(H_values_logmap_37), len(q_values))

# print(len(H_values_logmap_399), len(H_values_logmap_38), len(H_values_logmap_37), len(q_values))
plt.style.use('seaborn-darkgrid')
H_values_logmap_37 += [0] * (len(q_values)-len(H_values_logmap_37))
H_values_logmap_38 += [0] * (len(q_values)-len(H_values_logmap_38))
H_values_logmap_399 = [0]*(len(q_values)-len(H_values_logmap_399)) + H_values_logmap_399
print(len(H_values_logmap_399), len(H_values_logmap_38), len(H_values_logmap_37), len(q_values))
# Create a color palette
palette = plt.get_cmap('tab10')

points_to_mark = {
    "r = 3.7": [
        # (None, 0.2659, "True (30)"),
        # (None, 0.4872, "UKF (30)"),
        (0.38, 0.2094248118802859, "Optimal_logmap_37")
    ],
    "r = 3.8": [
        # (None, 0.1832, "True (22)"),
        # (None, 0.4467, "UKF (22)"),
        (0.38, 0.12435474122427852, "Optimal_logmap_38")
    ],
    "r = 3.99": [
        # (None, 0.1036, "True (17)"),
        # (None, 0.3380, "UKF (17)"),
        (1.1900000000000002, 0.299419305228393, "Optimal_logmap_399")
    ]
}

# Plot
plt.figure(figsize=(12, 8))

plt.plot(q_values, H_values_logmap_37, label='Logmap:r = 3.7')
plt.plot(q_values, H_values_logmap_38, label='Logmap:r = 3.8')
plt.plot(q_values, H_values_logmap_399, label='Logmap:r = 3.99')

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


