import numpy as np
import matplotlib.pyplot as plt
import json

joined = False

def adapt_y_values(original_q_values, original_y_values, common_q_values):
    adapted_y_values = np.empty_like(common_q_values)
    adapted_y_values[:] = np.NaN  # Fill with NaNs

    # Find indices in the common q_values array where original_q_values starts and ends
    start_idx = np.where(common_q_values >= original_q_values[0])[0][0]
    end_idx = start_idx + len(original_y_values)
    
    adapted_y_values[start_idx:end_idx] = original_y_values
    return adapted_y_values

if joined == "True":
    with open('plot_values.json') as json_file:
        data = json.load(json_file)

        mackey_arr1, mackey_arr2, mackey_arr3, mackey_arr4, mackey_arr5 = data["mackey_global_H_trace"][0], data["mackey_global_H_trace"][1], data["mackey_global_H_trace"][2], data["mackey_global_H_trace"][3], data["mackey_global_H_trace"][4]
        logmap_arr1, logmap_arr2, logmap_arr3, logmap_arr4, logmap_arr5 = data["logmap_global_H_trace"][0], data["logmap_global_H_trace"][1], data["logmap_global_H_trace"][2], data["logmap_global_H_trace"][3], data["logmap_global_H_trace"][4]
        optimal_points = {
        'Mackey:tau = 22, Realization 1': (data["mackey_frac_q_list"][0], data["mackey_frac_H_list"][0]),
        'Mackey:tau = 22, Realization 2': (data["mackey_frac_q_list"][1], data["mackey_frac_H_list"][1]),
        'Mackey:tau = 22, Realization 3': (data["mackey_frac_q_list"][2], data["mackey_frac_H_list"][2]),
        'Mackey:tau = 22, Realization 4': (data["mackey_frac_q_list"][3], data["mackey_frac_H_list"][3]),
        'Mackey:tau = 22, Realization 5': (data["mackey_frac_q_list"][4], data["mackey_frac_H_list"][4]),
        'Logmap:r = 3.9, Realization 1': (data["logmap_frac_q_list"][0], data["logmap_frac_H_list"][0]),
        'Logmap:r = 3.9, Realization 2': (data["logmap_frac_q_list"][1], data["logmap_frac_H_list"][1]),
        'Logmap:r = 3.9, Realization 3': (data["logmap_frac_q_list"][2], data["logmap_frac_H_list"][2]),
        'Logmap:r = 3.9, Realization 4': (data["logmap_frac_q_list"][3], data["logmap_frac_H_list"][3]),
        'Logmap:r = 3.9, Realization 5': (data["logmap_frac_q_list"][4], data["logmap_frac_H_list"][4]),
        }
        true_hurst_points = {
            'Mackey: tau = 22, Realization 1': [data["mackey_true_H_list"][0], '#1f77b4'],
            'Mackey: tau = 22, Realization 2': [data["mackey_true_H_list"][1], '#ff7f0e'],
            'Mackey: tau = 22, Realization 3': [data["mackey_true_H_list"][2], '#2ca02c'],
            'Mackey: tau = 22, Realization 4': [data["mackey_true_H_list"][3], '#d62728'],
            'Mackey: tau = 22, Realization 5': [data["mackey_true_H_list"][4], '#9467bd'],
            'Logmap:r = 3.9, Realization 1': [data["logmap_true_H_list"][0], '#f5f5dc'],
            'Logmap:r = 3.9, Realization 2': [data["logmap_true_H_list"][1], '#ffe4c4'],
            'Logmap:r = 3.9, Realization 3': [data["logmap_true_H_list"][2], '#000000'],
            'Logmap:r = 3.9, Realization 4': [data["logmap_true_H_list"][3], '#8a2be2'],
            'Logmap:r = 3.9, Realization 5': [data["logmap_true_H_list"][4], '#5f9ea0']
        }

        q_min, q_max = min(data["mackey_q_min"], data["logmap_q_min"]), max(data["mackey_q_max"], data["logmap_q_max"])
        q_values_common = np.arange(q_min, q_max , 0.01)
        # print(len(q_values_common))



        arr1_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), mackey_arr1, q_values_common)
        arr2_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), mackey_arr2, q_values_common)
        arr3_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), mackey_arr3, q_values_common)
        arr4_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), mackey_arr4, q_values_common)
        arr5_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), mackey_arr5, q_values_common)
        arr6_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), logmap_arr1, q_values_common)
        arr7_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), logmap_arr2, q_values_common)
        arr8_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), logmap_arr3, q_values_common)
        arr9_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), logmap_arr4, q_values_common)
        arr10_adapt = adapt_y_values(np.arange(q_min, q_max , 0.01), logmap_arr5, q_values_common)


        plt.figure(figsize=(15, 10))


        plots = [
            (q_values_common, arr1_adapt, 'Mackey:tau = 22, Realization 1', '--', 'o'),
            (q_values_common, arr2_adapt, 'Mackey:tau = 22, Realization 2', '--', 's'),
            (q_values_common, arr3_adapt, 'Mackey:tau = 22, Realization 3', '--', '^'),
            (q_values_common, arr4_adapt, 'Mackey:tau = 22, Realization 4', '--', 'x'),
            (q_values_common, arr5_adapt, 'Mackey:tau = 22, Realization 5', '--', '+'),
            (q_values_common, arr6_adapt, 'Logmap:r = 3.9, Realization 1', ':', 'o'),
            (q_values_common, arr7_adapt, 'Logmap:r = 3.9, Realization 2', ':', 's'),
            (q_values_common, arr8_adapt, 'Logmap:r = 3.9 Realization 3', ':', '^'),
            (q_values_common, arr9_adapt, 'Logmap:r = 3.9, Realization 4', ':', 'x'),
            (q_values_common, arr10_adapt, 'Logmap:r = 3.9, Realization 5', ':', '+')
        ]

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
        max_x_value = q_max + 0.1
        plt.axvline(x=max_x_value, color='grey', linestyle='--')
        # Plot the true Hurst exponents on the vertical line
        for label, h in true_hurst_points.items():
            ax.scatter(max_x_value, h[0], color=h[1], zorder=5, marker = 's')
            ax.text(max_x_value, h[0], f'  H={h[0]}', verticalalignment='center')
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


def plot_trace(H_lol, optimal_q_list, optimal_H_list, true_list, q_min, q_max):
 
    arr1, arr2, arr3, arr4, arr5 = H_lol[0], H_lol[1], H_lol[2], H_lol[3], H_lol[4]
    optimal_points = {
    'Mackey:tau = 17,  1': (optimal_q_list[0], optimal_H_list[0]),
    'Mackey:tau = 17, 2': (optimal_q_list[1], optimal_H_list[1]),
    'Mackey:tau = 17, 3': (optimal_q_list[2], optimal_H_list[2]),
    'Mackey:tau = 17, 4': (optimal_q_list[3], optimal_H_list[3]),
    'Mackey:tau = 17, 5': (optimal_q_list[4], optimal_H_list[4])

    }

    true_hurst_points = {
        'Mackey: tau = 17, 1': [true_list[0], '#1f77b4'],
        'Mackey: tau = 17, 2': [true_list[1], '#ff7f0e'],
        'Mackey: tau = 17, 3': [true_list[2], '#2ca02c'],
        'Mackey: tau = 17, 4': [true_list[3], '#d62728'],
        'Mackey: tau = 17, 5': [true_list[4], '#9467bd']
    }

    # Create a common q_values array
    q_values_common = np.arange(q_min, q_max, 0.01)
    print(len(q_values_common))



    arr1_adapt = adapt_y_values(np.arange(q_min, q_max, 0.01), arr1, q_values_common)
    arr2_adapt = adapt_y_values(np.arange(q_min, q_max, 0.01), arr2, q_values_common)
    arr3_adapt = adapt_y_values(np.arange(q_min, q_max, 0.01), arr3, q_values_common)
    arr4_adapt = adapt_y_values(np.arange(q_min, q_max, 0.01), arr4, q_values_common)
    arr5_adapt = adapt_y_values(np.arange(q_min, q_max, 0.01), arr5, q_values_common)


    plt.figure(figsize=(15, 10))


    plots = [
        (q_values_common, arr1_adapt, 'Mackey:tau = 17, 1', '-', 'o'),
        (q_values_common, arr2_adapt, 'Mackey:tau = 17, 2', '-', 's'),
        (q_values_common, arr3_adapt, 'Mackey:tau = 17, 3', '-', '^'),
        (q_values_common, arr4_adapt, 'Mackey:tau = 17, 4', ':', 'x'),
        (q_values_common, arr5_adapt, 'Mackey:tau = 17, 5', ':', '+'),
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
    max_x_value = q_max + 0.1
    plt.axvline(x=max_x_value, color='grey', linestyle='--')
    # Plot the true Hurst exponents on the vertical line
    for label, h in true_hurst_points.items():
        ax.scatter(max_x_value, h[0], color=h[1], zorder=5, marker = 's')
        ax.text(max_x_value, h[0], f'  H={h[0]}', verticalalignment='center')
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


def adapt_y_values(original_q_values, original_y_values, common_q_values):
    adapted_y_values = np.empty_like(common_q_values)
    adapted_y_values[:] = np.NaN  # Fill with NaNs

    # Find indices in the common q_values array where original_q_values starts and ends
    start_idx = np.where(common_q_values >= original_q_values[0])[0][0]
    end_idx = start_idx + len(original_y_values)
    
    adapted_y_values[start_idx:end_idx] = original_y_values
    return adapted_y_values

