import numpy as np
import matplotlib.pyplot as plt

# ... [Define all H_values and points_to_mark for all your plots here]
H_values_mackey_30 = [0.3727,0.3714,0.3702,0.3689,0.3677,0.3664,0.3652,0.3639,0.3626,0.3614,0.3601,0.3589,0.3576,0.3563,0.3551,0.3538,0.3525,0.3513,0.3500,0.3487,0.3474,0.3462,0.3449,0.3436,0.3424,0.3411,0.3398,0.3386,0.3373,0.3360,0.3348,0.3335,0.3323,0.3310,0.3297,0.3285,0.3272,0.3260,0.3247,0.3235,0.3222,0.3210,0.3197,0.3185,0.3172,0.3160,0.3147,0.3135,0.3123,0.3110,0.3098,0.3086,0.3073,0.3061,0.3049,0.3036,0.3024,0.3012,0.3000,0.2987,0.2975,0.2963,0.2951,0.2939,0.2927,0.2915,0.2903,0.2891,0.2879,0.2866,0.2856,0.2845,0.2835,0.2824,0.2814,0.2804,0.2794,0.2783,0.2773,0.2763,0.2753,0.2742,0.2732,0.2722,0.2712,0.2701,0.2691,0.2681,0.2671,0.2661,0.2651,0.2642,0.2632,0.2623,0.2613,0.2604,0.2595,0.2585,0.2576,0.2567] # Fill in with all the H values for tau = 30
H_values_mackey_22 = [0.3300, 0.3287, 0.3273, 0.3260, 0.3247, 0.3233, 0.3220, 0.3207, 0.3193, 0.3180, 0.3166, 0.3153, 0.3140, 0.3126, 0.3113, 0.3100, 0.3086, 0.3074, 0.3062, 0.3050, 0.3038, 0.3026, 0.3015, 0.3003, 0.2991, 0.2979, 0.2967, 0.2955, 0.2943, 0.2931, 0.2919, 0.2907, 0.2895, 0.2883, 0.2871, 0.2859, 0.2847, 0.2835, 0.2823, 0.2811, 0.2800, 0.2788, 0.2777, 0.2765, 0.2754, 0.2742, 0.2731, 0.2719, 0.2708, 0.2696, 0.2685, 0.2674, 0.2662, 0.2651, 0.2639, 0.2628, 0.2617, 0.2605, 0.2594, 0.2583, 0.2571, 0.2560, 0.2548, 0.2538, 0.2528, 0.2518, 0.2508, 0.2498, 0.2488, 0.2478, 0.2468, 0.2458, 0.2449, 0.2439, 0.2429, 0.2419, 0.2409, 0.2399, 0.2389, 0.2379, 0.2369, 0.2360, 0.2350, 0.2340, 0.2330, 0.2320, 0.2311, 0.2316, 0.2325, 0.2334, 0.2342, 0.2351, 0.2360, 0.2368, 0.2377, 0.2386, 0.2394, 0.2403, 0.2411, 0.2420] # Fill in with all the H values for tau = 22
H_values_mackey_17 = [0.2104, 0.2091, 0.2078, 0.2065, 0.2051, 0.2038, 0.2025, 0.2012, 0.1999, 0.1986, 0.1973, 0.1959, 0.1946, 0.1933, 0.1920, 0.1907, 0.1894, 0.1880, 0.1867, 0.1854, 0.1841, 0.1828, 0.1815, 0.1802, 0.1789, 0.1776, 0.1763, 0.1750, 0.1737, 0.1725, 0.1712, 0.1699, 0.1686, 0.1673, 0.1661, 0.1648, 0.1636, 0.1625, 0.1614, 0.1602, 0.1601, 0.1611, 0.1620, 0.1630, 0.1640, 0.1649, 0.1658, 0.1668, 0.1677, 0.1686, 0.1696, 0.1705, 0.1714, 0.1723, 0.1732, 0.1741, 0.1750, 0.1759, 0.1768, 0.1777, 0.1786, 0.1794, 0.1803, 0.1812, 0.1821, 0.1829, 0.1838, 0.1846, 0.1855, 0.1863, 0.1872, 0.1880, 0.1889, 0.1897, 0.1906, 0.1914, 0.1922, 0.1931, 0.1939, 0.1947, 0.1955, 0.1964, 0.1972, 0.1980, 0.1988, 0.1996, 0.2004, 0.2012, 0.2020, 0.2028, 0.2036, 0.2044, 0.2052, 0.2060, 0.2069, 0.2077, 0.2085, 0.2093, 0.2101, 0.2109] # Fill in with all the H values for tau = 17
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
H_values_tentmap_199 = [0.4244, 
0.4328, 
0.4409, 
0.4487, 
0.4562, 
0.4634, 
0.4702, 
0.4769, 
0.4832, 
0.4893, 
0.4951, 
0.5007, 
0.5059, 
0.5110, 
0.5158, 
0.5204, 
0.5247, 
0.5288, 
0.5329, 
0.5369, 
0.5405, 
0.5434, 
0.5460, 
0.5483, 
0.5497, 
0.5503, 
0.5511, 
0.5547, 
0.5580, 
0.5606, 
0.5623, 
0.5635, 
0.5645, 
0.5652, 
0.5656, 
0.5658, 
0.5657, 
0.5654, 
0.5648, 
0.5639, 
0.5628, 
0.5613, 
0.5596, 
0.5577, 
0.5555, 
0.5530, 
0.5500, 
0.5469, 
0.5434, 
0.5397, 
0.5357, 
0.5316, 
0.5273, 
0.5228, 
0.5181, 
0.5132, 
0.5081, 
0.5029, 
0.4975, 
0.4929, 
0.4884, 
0.4839, 
0.4793, 
0.4747, 
0.4701, 
0.4655, 
0.4609, 
0.4563, 
0.4518, 
0.4473, 
0.4428, 
0.4383, 
0.4340, 
0.4297, 
0.4255, 
0.4213, 
0.4173, 
0.4133, 
0.4094, 
0.4056, 
0.4020, 
0.3984, 
0.3949, 
0.3915, 
0.3883, 
0.3851, 
0.3820, 
0.3790, 
0.3761, 
0.3741, 
0.3761, 
0.3781, 
0.3803, 
0.3825, 
0.3848, 
0.3872, 
0.3896, 
0.3921, 
0.3946, 
0.3972, 
0.3998, 
0.4025, 
0.4052, 
0.4079, 
0.4106, 
0.4134, 
0.4162, 
0.4189, 
0.4217, 
0.4245, 
0.4273, 
0.4300, 
0.4328, 
0.4355, 
0.4382, 
0.4409, 
0.4436, 
0.4462, 
0.4488, 
0.4514, 
0.4539, 
0.4564, 
0.4591, 
0.4618, 
0.4645, 
0.4671, 
0.4697, 
0.4722, 
0.4747, 
0.4771, 
0.4787, 
0.4801, 
0.4814, 
0.4827, 
0.4839, 
0.4851, 
0.4863, 
0.4874, 
0.4885, 
0.4895, 
0.4905, 
0.4915, 
0.4924, 
0.4932, 
0.4940, 
0.4948, 
0.4955, 
0.4961, 
0.4965, 
0.4969, 
0.4973, 
0.4977, 
0.4980, 
0.4983, 
0.4986, 
0.4988, 
0.4991, 
0.4993, 
0.4995, 
0.4996, 
0.4997, 
0.4999, 
0.4999, 
0.5000, 
0.5001, 
0.5001, 
0.5000, 
0.5000, 
0.4999, 
0.4999, 
0.4998, 
0.4996, 
0.4995, 
0.4993, 
0.4991, 
0.4989, 
0.4987, 
0.4984, 
0.4982, 
0.4979, 
0.4976, 
0.4972, 
0.4969, 
0.4965, 
0.4961, 
0.4957, 
0.4953, 
0.4948, 
0.4944, 
0.4966, 
0.5011, 
0.5057, 
0.5102, 
0.5148, 
0.5193, 
0.5239, 
0.5285, 
0.5325, 
0.5366, 
0.5407] # Fill in with all the H values for tau = 30
H_values_tentmap_195 = [0.9249, 
0.9309, 
0.9365, 
0.9420, 
0.9475, 
0.9528, 
0.9579, 
0.9630, 
0.9680, 
0.9729, 
0.9776, 
0.9823, 
0.9868, 
0.9913, 
0.9956, 
0.9986, 
1.0012, 
1.0038, 
1.0063, 
1.0085, 
1.0103, 
1.0120, 
1.0136, 
1.0151, 
1.0161, 
1.0164, 
1.0167, 
1.0170, 
1.0172, 
1.0173, 
1.0173, 
1.0172, 
1.0170, 
1.0167, 
1.0163, 
1.0159, 
1.0153, 
1.0146, 
1.0139, 
1.0130, 
1.0120, 
1.0109, 
1.0096, 
1.0082, 
1.0067, 
1.0051, 
1.0032, 
1.0012, 
0.9991, 
0.9968, 
0.9943, 
0.9916, 
0.9887, 
0.9857, 
0.9824, 
0.9788, 
0.9749, 
0.9709, 
0.9667, 
0.9622, 
0.9576, 
0.9527, 
0.9476, 
0.9424, 
0.9369, 
0.9312, 
0.9254, 
0.9193, 
0.9131, 
0.9067, 
0.9002, 
0.8935, 
0.8867, 
0.8797, 
0.8726, 
0.8654, 
0.8580, 
0.8506, 
0.8431, 
0.8356, 
0.8280, 
0.8203, 
0.8126, 
0.8049, 
0.7972, 
0.7895, 
0.7817, 
0.7740, 
0.7662, 
0.7585, 
0.7508, 
0.7430, 
0.7352, 
0.7274, 
0.7196, 
0.7118, 
0.7040, 
0.6963, 
0.6885, 
0.6808, 
0.6730, 
0.6652, 
0.6575, 
0.6497, 
0.6420, 
0.6343, 
0.6265, 
0.6188, 
0.6111, 
0.6034, 
0.5957, 
0.5880, 
0.5802, 
0.5726, 
0.5649, 
0.5572, 
0.5494, 
0.5416, 
0.5338, 
0.5260, 
0.5182, 
0.5104, 
0.5089, 
0.5144, 
0.5199, 
0.5253, 
0.5307, 
0.5360, 
0.5413, 
0.5445, 
0.5472, 
0.5498, 
0.5523, 
0.5548, 
0.5573, 
0.5599, 
0.5624, 
0.5650, 
0.5676, 
0.5702, 
0.5728, 
0.5753, 
0.5778, 
0.5803, 
0.5828, 
0.5854, 
0.5878, 
0.5903, 
0.5928, 
0.5952, 
0.5974, 
0.5996, 
0.6018, 
0.6040, 
0.6062, 
0.6085, 
0.6106, 
0.6127, 
0.6148, 
0.6170, 
0.6194, 
0.6219, 
0.6243, 
0.6268, 
0.6292, 
0.6317, 
0.6342, 
0.6368, 
0.6393, 
0.6419, 
0.6445, 
0.6471, 
0.6497, 
0.6524, 
0.6551, 
0.6577, 
0.6604, 
0.6631, 
0.6659, 
0.6686, 
0.6713, 
0.6740, 
0.6768, 
0.6795, 
0.6823, 
0.6850, 
0.6877, 
0.6904, 
0.6931, 
0.6958, 
0.6985, 
0.7013, 
0.7040, 
0.7066, 
0.7092, 
0.7117, 
0.7141, 
0.7163, 
0.7186, 
0.7209] # Fill in with all the H values for tau = 22
H_values_tentmap_19 = [0.3211, 
0.3209, 
0.3206, 
0.3202, 
0.3198, 
0.3194, 
0.3189, 
0.3183, 
0.3178, 
0.3172, 
0.3167, 
0.3161, 
0.3153, 
0.3145, 
0.3137, 
0.3127, 
0.3124, 
0.3190, 
0.3254, 
0.3319, 
0.3383, 
0.3446, 
0.3509, 
0.3571, 
0.3634, 
0.3695, 
0.3757, 
0.3817, 
0.3878, 
0.3938, 
0.3998, 
0.4057, 
0.4116, 
0.4175, 
0.4233, 
0.4291, 
0.4349, 
0.4407, 
0.4464, 
0.4521, 
0.4577, 
0.4634, 
0.4690, 
0.4746, 
0.4802, 
0.4858, 
0.4913, 
0.4968, 
0.5023, 
0.5078, 
0.5133, 
0.5188, 
0.5240, 
0.5285, 
0.5329, 
0.5373, 
0.5417, 
0.5462, 
0.5506, 
0.5549, 
0.5589, 
0.5629, 
0.5669, 
0.5709, 
0.5750, 
0.5788, 
0.5826, 
0.5864, 
0.5901, 
0.5938, 
0.5976, 
0.6012, 
0.6047, 
0.6083, 
0.6118, 
0.6154, 
0.6188, 
0.6221, 
0.6255, 
0.6290, 
0.6325, 
0.6360, 
0.6396, 
0.6430, 
0.6464, 
0.6497, 
0.6531, 
0.6565, 
0.6599, 
0.6631, 
0.6651, 
0.6671, 
0.6690, 
0.6706, 
0.6723, 
0.6741, 
0.6758, 
0.6775, 
0.6791, 
0.6807, 
0.6823, 
0.6839, 
0.6856, 
0.6875, 
0.6896, 
0.6917, 
0.6939, 
0.6960, 
0.6982, 
0.7004, 
0.7027, 
0.7049, 
0.7071, 
0.7093, 
0.7115, 
0.7138, 
0.7160, 
0.7180, 
0.7200, 
0.7221, 
0.7241, 
0.7261, 
0.7282, 
0.7302, 
0.7322, 
0.7343, 
0.7364, 
0.7384, 
0.7403, 
0.7422, 
0.7442, 
0.7462, 
0.7482, 
0.7502, 
0.7520, 
0.7538, 
0.7556, 
0.7574, 
0.7592, 
0.7610, 
0.7629, 
0.7647, 
0.7665, 
0.7684, 
0.7702, 
0.7720, 
0.7739, 
0.7757, 
0.7776, 
0.7795, 
0.7814, 
0.7832, 
0.7851, 
0.7869, 
0.7888, 
0.7907, 
0.7926, 
0.7945, 
0.7964, 
0.7984, 
0.8003, 
0.8021, 
0.8039, 
0.8057, 
0.8075, 
0.8093, 
0.8111, 
0.8129, 
0.8147, 
0.8166, 
0.8184, 
0.8202, 
0.8221, 
0.8239, 
0.8257, 
0.8276, 
0.8294, 
0.8313, 
0.8332, 
0.8351, 
0.8370, 
0.8389, 
0.8408, 
0.8427, 
0.8446, 
0.8466, 
0.8485, 
0.8505, 
0.8525, 
0.8544, 
0.8564, 
0.8584, 
0.8604, 
0.8624, 
0.8644, 
0.8665, 
0.8685, 
0.8705, 
0.8726, 
0.8747] # Fill in with all the H values for tau = 17
optimal_points = {
    'Mackey:tau = 30': (1.9800000000000002, 0.26710019697747805),
    'Mackey:tau = 22': (1.9600000000000009, 0.2311101372511777),
    'Mackey:tau = 17': (1.5000000000000002, 0.16010560316294808),
    'Logmap:r = 3.7': (0.38, 0.2094248118802859),
    'Logmap:r = 3.8': (0.38, 0.12435474122427852),
    'Logmap:r = 3.9': (1.1900000000000002, 0.299419305228393),
    'TentMap:mu = 1.99': (1.9900000000000009, 0.3740775278132317),
    'TentMap:mu = 1.95': (2.320000000000001, 0.5089346216984104),
    'TentMap:mu = 1.9': (1.2600000000000002, 0.31244267910500617),
}
true_hurst_points = {
    'Mackey: tau = 30': [0.2659, '#1f77b4'],
    'tau = 22': [0.1832, '#ff7f0e'],
    'tau = 17': [0.1036, '#2ca02c'],
    'Logmap: r = 3.7': [0.0837, '#d62728'],
    'r = 3.8': [0.0742, '#9467bd'],
    'r = 3.99': [0.4718, '#8c564b'],
    'Tentmap: mu = 1.99': [0.3719, '#e377c2'],
    'mu = 1.95': [0.4561, '#7f7f7f'],
    'mu = 1.9': [0.3075, '#bcbd22'],
}

# Create a common q_values array
q_values_common = np.arange(0, 5.01, 0.01)

# Adapt each y-values array to the common q_values array. 
# For this example, I'm using your provided q_values array for H_values_tentmap_199
# The other H_values arrays should be adapted similarly
def adapt_y_values(original_q_values, original_y_values, common_q_values):
    adapted_y_values = np.empty_like(common_q_values)
    adapted_y_values[:] = np.NaN  # Fill with NaNs

    # Find indices in the common q_values array where original_q_values starts and ends
    start_idx = np.where(common_q_values >= original_q_values[0])[0][0]
    end_idx = start_idx + len(original_y_values)
    
    adapted_y_values[start_idx:end_idx] = original_y_values
    return adapted_y_values

H_values_mackey_30_adapted = adapt_y_values(np.arange(1.1, 2.1, 0.01), H_values_mackey_30, q_values_common)
H_values_mackey_22_adapted = adapt_y_values(np.arange(1.1, 2.1, 0.01), H_values_mackey_22, q_values_common)
H_values_mackey_17_adapted = adapt_y_values(np.arange(1.1, 2.1, 0.01), H_values_mackey_17, q_values_common)
H_values_logmap_37_adapted = adapt_y_values(np.arange(0.01, 0.99, 0.01), H_values_logmap_37, q_values_common)
H_values_logmap_38_adapted = adapt_y_values(np.arange(0.01, 0.99, 0.01), H_values_logmap_38, q_values_common)
H_values_logmap_399_adapted = adapt_y_values(np.arange(1.1, 1.5, 0.01), H_values_logmap_37, q_values_common)
H_values_tentmap_199_adapted = adapt_y_values(np.arange(1.1, 3.1, 0.01), H_values_tentmap_199, q_values_common)
H_values_tentmap_195_adapted = adapt_y_values(np.arange(1.1, 3.1, 0.01), H_values_tentmap_195, q_values_common)
H_values_tentmap_19_adapted = adapt_y_values(np.arange(1.1, 3.1, 0.01), H_values_tentmap_19, q_values_common)
# ... [Adapt other H_values arrays similarly]

plt.figure(figsize=(15, 10))

# Define the plot configurations
plots = [
    (q_values_common, H_values_mackey_30_adapted, 'Mackey:tau = 30', '-', 'o'),
    (q_values_common, H_values_mackey_22_adapted, 'Mackey:tau = 22', '-', 's'),
    (q_values_common, H_values_mackey_17_adapted, 'Mackey:tau = 17', '-', '^'),
    (q_values_common, H_values_logmap_37_adapted, 'Logmap:r = 3.7', ':', 'x'),
    (q_values_common, H_values_logmap_38_adapted, 'Logmap:r = 3.8', ':', '+'),
    (q_values_common, H_values_logmap_399_adapted, 'Logmap:r = 3.99', ':', '*'),
    (q_values_common, H_values_tentmap_199_adapted, 'TentMap:mu = 1.99', '--', 'D'),
    (q_values_common, H_values_tentmap_195_adapted, 'TentMap:mu = 1.95', '--', 'P'),
    (q_values_common, H_values_tentmap_19_adapted, 'TentMap:mu = 1.9', '--', 'H'),
]

# Store colors to use for optimal values later
colors = []

# Plotting
for q_vals, H_vals, label, linestyle, marker in plots:
    line, = plt.plot(q_vals, H_vals, label=label, linestyle=linestyle, marker=marker, markevery=20)
    colors.append(line.get_color())

# Additional Plot Settings
plt.xlabel('q values', fontsize=14)
plt.ylabel('Hurst Exponent (H)', fontsize=14)
plt.title('Hurst Exponent (H) vs. q values', fontsize=16)

# Marking optimal points with corresponding colors
for idx, (label, (q, h)) in enumerate(optimal_points.items()):
    plt.scatter(q, h, color=colors[idx], s=150, zorder=5)  # zorder to make sure it's on top
ax = plt.gca()
max_x_value = 3.11
plt.axvline(x=max_x_value, color='grey', linestyle='--')
# Plot the true Hurst exponents on the vertical line
for label, h in true_hurst_points.items():
    ax.scatter(max_x_value, h[0], color=h[1], zorder=5, marker = 's')
    ax.text(max_x_value, h[0], f'H={h[0]}', verticalalignment='center')
    ax.axhline(y=h[0], xmin=0, xmax=max_x_value / (max_x_value + 0.1 * (max_x_value - min(q_values_common))), color=h[1], linestyle=':', linewidth=1, alpha=1)

# Draw the vertical line at the end of the x-axis
plt.axvline(x=max_x_value, color='grey', linestyle='--')

# Adjust the x-axis limits to make space for the text
plt.xlim(min(q_values_common), max_x_value + 0.1 * (max_x_value - min(q_values_common)))

# Adjusting grid aesthetics
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.6)
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)

# Legend specification
# Add a black marker for the optimal value indication in the legend
hurst_legend = plt.Line2D([0], [0], marker='o', color='w', label='Fractal UKF Optimal H', markersize=10, markerfacecolor='cyan')
optimal_legend = plt.Line2D([0], [0], marker='s', color='w', label='True H', markersize=10, markerfacecolor='cyan')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(hurst_legend)
handles.append(optimal_legend)
plt.legend(handles=handles, fontsize=10, loc="best")

# Show the plot
plt.tight_layout()
plt.show()





