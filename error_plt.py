import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

cases = ["mackey_30", "mackey_22", "mackey_17", "logmap_3.7", "logmap_3.8", "logmap_3.99", "tent_1.99", "tent_1.95", "tent_1.9"]
# cases = ["mackey_30", "mackey_22", "mackey_17"]
means_ukf = [4.2829638882176385e-07, 4.955107220569403e-07, 6.116290103801713e-06, 0.00014948570078134778, 0.00017207259922602937, 5.551552475035557e-05, 3.0218440474310252e-05, 0.00010456428949736716, 0.00012201683982460373]
means_fractal_ukf = [2.4419514229816258e-08, 1.6874262289320964e-06, 1.5977370085858736e-07, 8.412318867371236e-05, 7.225867061293865e-05, 3.611833862569719e-05, 4.506975484314761e-06, 9.28762706731393e-05, 7.321961323876157e-05]
variances_ukf = [0.006146225693944442, 0.007175239106977528, 0.007922860463454955, 0.006143763008750899, 0.006997926240875862, 0.009021599475729215, 0.009090980724925255, 0.008449441459506333, 0.008464440789215613]
variances_fractal_ukf = [0.011259237205881508, 0.01298330221886977, 0.012326025084660457, 0.02218061603540751, 0.02149157502725528, 0.08187670657572173, 0.07687991625525611, 0.06523104434002912, 0.0559910067088932]
# means_ukf = [4.2829638882176385e-07, 4.955107220569403e-07, 6.116290103801713e-06]
# means_fractal_ukf = [2.4419514229816258e-08, 1.6874262289320964e-06, 1.5977370085858736e-07]
# variances_ukf = [0.006146225693944442, 0.007175239106977528, 0.007922860463454955]
# variances_fractal_ukf = [0.011259237205881508, 0.01298330221886977, 0.012326025084660457]

# Calculating NMSE for UKF
nmse_ukf = [mse / variance for mse, variance in zip(means_ukf, variances_ukf)]

# Calculating NMSE for Fractal UKF
nmse_fractal_ukf = [mse / variance for mse, variance in zip(means_fractal_ukf, variances_fractal_ukf)]

print("NMSE for UKF:", nmse_ukf)
print("NMSE for Fractal UKF:", nmse_fractal_ukf)

fig, ax = plt.subplots(figsize=(14, 8))

# Font customization
plt.rcParams["font.family"] = "Arial"

# Using errorbar for UKF
x_pos_ukf = np.arange(len(cases)) - 0.2
ax.errorbar(x_pos_ukf, nmse_ukf, yerr=variances_ukf, fmt='o', color='#1f77b5', label='UKF', capsize=5, capthick=1.5, elinewidth=1.5)

# Using errorbar for Fractal UKF
x_pos_fractal_ukf = np.arange(len(cases)) + 0.2
ax.errorbar(x_pos_fractal_ukf, nmse_fractal_ukf, yerr=variances_fractal_ukf, fmt='o', color='#d62728', label='Fractal UKF', capsize=5, capthick=1.5, elinewidth=1.5)

# Box each column separately by adding a Rectangle patch
box_width = 0.5  # Width of the rectangle
for i in range(len(cases)):
    rect = patches.Rectangle((i - box_width / 2, ax.get_ylim()[0]), box_width, ax.get_ylim()[1] - ax.get_ylim()[0], linewidth=0, edgecolor=None, facecolor='#f0f0f0', zorder=0)
    ax.add_patch(rect)

# Background styling
ax.set_facecolor("#f5f5f5") # Light gray background
ax.grid(axis="y", color="white", linewidth=2)
for spine in ax.spines.values():
    spine.set_visible(False)

# Setting axis labels, title, and legend
ax.set_xticks(np.arange(len(cases)))
ax.set_xticklabels(cases, rotation=45)
ax.set_title('Error Bars for NMSE and Estimation Variance', fontsize=16)
ax.set_ylabel('Value', fontsize=14)

# Y-axis formatting
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
ax.yaxis.get_offset_text().set_fontsize(12) # Adjust fontsize of the exponent
ax.tick_params(axis='both', which='major', labelsize=12)

ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()


# # fig, ax = plt.subplots(figsize=(14, 8))

# # # Error bar for UKF
# # x_pos_ukf = np.arange(len(cases)) - 0.2
# # ax.errorbar(x_pos_ukf, nmse_ukf, fmt='o', color='blue', label='UKF NMSE', capsize=5)

# # # Error bar for Fractal UKF
# # x_pos_fractal_ukf = np.arange(len(cases)) + 0.2
# # ax.errorbar(x_pos_fractal_ukf, nmse_fractal_ukf, fmt='o', color='red', label='Fractal UKF NMSE', capsize=5)

# # # Setting axis labels, title, and legend
# # ax.set_xticks(np.arange(len(cases)))
# # ax.set_xticklabels(cases, rotation=45)
# # ax.set_title('Normalized Mean Squared Errors')
# # ax.set_ylabel('NMSE Value')
# # ax.legend()

# # plt.tight_layout()
# # plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.patches as patches

# cases = ["mackey_30", "mackey_22", "mackey_17", "logmap_3.7", "logmap_3.8", "logmap_3.99", "tent_1.99", "tent_1.95", "tent_1.9"]

# # Calculating NMSE
# nmse_ukf = [mse / variance for mse, variance in zip(means_ukf, variances_ukf)]
# nmse_fractal_ukf = [mse / variance for mse, variance in zip(means_fractal_ukf, variances_fractal_ukf)]

# fig, ax = plt.subplots(figsize=(14, 8))

# # Font customization
# plt.rcParams["font.family"] = "Arial"

# # Using errorbar for UKF NMSE
# x_pos_ukf = np.arange(len(cases)) - 0.2
# ax.errorbar(x_pos_ukf, nmse_ukf, yerr=variances_ukf, fmt='o', color='#1f77b5', label='UKF NMSE', capsize=5, capthick=1.5, elinewidth=1.5)

# # Using errorbar for Fractal UKF NMSE
# x_pos_fractal_ukf = np.arange(len(cases)) + 0.2
# ax.errorbar(x_pos_fractal_ukf, nmse_fractal_ukf, yerr=variances_fractal_ukf, fmt='o', color='#d62728', label='Fractal UKF NMSE', capsize=5, capthick=1.5, elinewidth=1.5)

# # Box each column separately by adding a Rectangle patch
# box_width = 0.5  # Width of the rectangle
# for i in range(len(cases)):
#     rect = patches.Rectangle((i - box_width / 2, ax.get_ylim()[0]), box_width, ax.get_ylim()[1] - ax.get_ylim()[0], linewidth=0, edgecolor=None, facecolor='#f0f0f0', zorder=0)
#     ax.add_patch(rect)

# # Background styling
# ax.set_facecolor("#f5f5f5") # Light gray background
# ax.grid(axis="y", color="white", linewidth=2)
# for spine in ax.spines.values():
#     spine.set_visible(False)

# # Setting axis labels, title, and legend
# ax.set_xticks(np.arange(len(cases)))
# ax.set_xticklabels(cases, rotation=45)
# ax.set_title('Normalized Mean Squared Errors', fontsize=16, fontweight='bold')
# ax.set_ylabel('NMSE Value', fontsize=14)

# # Y-axis formatting
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
# ax.yaxis.get_offset_text().set_fontsize(12) # Adjust fontsize of the exponent
# ax.tick_params(axis='both', which='major', labelsize=12)

# ax.legend(loc='upper left', fontsize=12)

# plt.tight_layout()
# plt.show()
