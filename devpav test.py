from functions_thesis import *
from math import sqrt, comb
import math
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize mean and variance, amount of data and scaling factors
MU_ss = 6
SIGMA = sqrt(2 * MU_ss)
n_ss = 150
n_sd = 450
c = 1
c_2 = 2
d = 1.5
d_2 = 2.5

# Initialize amount of times to calculate metrics
N = 1000

# Initialize arrays to store normal devPAV values
dev_p = []
dev_r1 = []
dev_r2 = []
dev_l1 = []
dev_l2 = []
dev_e1 = []
dev_e2 = []
dev_w1 = []
dev_w2 = []

# Initialize arrays to store scaled devPAV values
sdev_p = []
sdev_r1 = []
sdev_r2 = []
sdev_l1 = []
sdev_l2 = []
sdev_e1 = []
sdev_e2 = []
sdev_w1 = []
sdev_w2 = []

# Initialize arrays to store smoothed devPAV values
cdev_p = []
cdev_r1 = []
cdev_r2 = []
cdev_l1 = []
cdev_l2 = []
cdev_e1 = []
cdev_e2 = []
cdev_w1 = []
cdev_w2 = []

for i in range(N):
    try:
        # Generate consistent data
        LSS_p = np.random.normal(MU_ss, SIGMA, n_ss)
        LDS_p = np.random.normal(-MU_ss, SIGMA, n_sd)
        SS_p = np.power(math.e, LSS_p)
        DS_p = np.power(math.e, LDS_p)

        # Generate data skewed to the right by c and c_2
        LSS_r1 = LSS_p + c
        LDS_r1 = LDS_p + c
        SS_r1 = np.power(math.e, LSS_r1)
        DS_r1 = np.power(math.e, LDS_r1)

        LSS_r2 = LSS_p + c_2
        LDS_r2 = LDS_p + c_2
        SS_r2 = np.power(math.e, LSS_r2)
        DS_r2 = np.power(math.e, LDS_r2)

        # Generate data skewed to left by c and c_2
        LSS_l1 = LSS_p - c
        LDS_l1 = LDS_p - c
        SS_l1 = np.power(math.e, LSS_l1)
        DS_l1 = np.power(math.e, LDS_l1)

        LSS_l2 = LSS_p - c_2
        LDS_l2 = LDS_p - c_2
        SS_l2 = np.power(math.e, LSS_l2)
        DS_l2 = np.power(math.e, LDS_l2)

        # Generate too extreme data, scaled by d and d_2
        LSS_e1 = d * LSS_p
        LDS_e1 = d * LDS_p
        SS_e1 = np.power(math.e, LSS_e1)
        DS_e1 = np.power(math.e, LDS_e1)

        LSS_e2 = d_2 * LSS_p
        LDS_e2 = d_2 * LDS_p
        SS_e2 = np.power(math.e, LSS_e2)
        DS_e2 = np.power(math.e, LDS_e2)

        # Generate too weak data, scaled by d and d_2
        LSS_w1 = (1 / d) * LSS_p
        LDS_w1 = (1 / d) * LDS_p
        SS_w1 = np.power(math.e, LSS_w1)
        DS_w1 = np.power(math.e, LDS_w1)

        LSS_w2 = (1 / d_2) * LSS_p
        LDS_w2 = (1 / d_2) * LDS_p
        SS_w2 = np.power(math.e, LSS_w2)
        DS_w2 = np.power(math.e, LDS_w2)

        # Determine metrics for consistent data
        lrs = np.concatenate((SS_p, DS_p))
        all_hypotheses = np.concatenate((np.array(['H1'] * len(SS_p)), np.array(['H2'] * len(DS_p))))
        all_hypotheses_01 = np.where(all_hypotheses == 'H1', 1, 0)

        dp_p = devpav(lrs, all_hypotheses_01)
        dp_p1 = scaled_devpav(lrs, all_hypotheses_01)
        dp_p2 = devpav_new(lrs, all_hypotheses_01)

        # Determine metrics for data skewed to the right
        lrs = np.concatenate((SS_r1, DS_r1))

        dp_r1 = devpav(lrs, all_hypotheses_01)
        dp_r11 = scaled_devpav(lrs, all_hypotheses_01)
        dp_r12 = devpav_new(lrs, all_hypotheses_01)

        lrs = np.concatenate((SS_r2, DS_r2))

        dp_r2 = devpav(lrs, all_hypotheses_01)
        dp_r21 = scaled_devpav(lrs, all_hypotheses_01)
        dp_r22 = devpav_new(lrs, all_hypotheses_01)

        # Determine metrics for data skewed to the left
        lrs = np.concatenate((SS_l1, DS_l1))

        dp_l1 = devpav(lrs, all_hypotheses_01)
        dp_l11 = scaled_devpav(lrs, all_hypotheses_01)
        dp_l12 = devpav_new(lrs, all_hypotheses_01)

        lrs = np.concatenate((SS_l2, DS_l2))
        dp_l2 = devpav(lrs, all_hypotheses_01)
        dp_l21 = scaled_devpav(lrs, all_hypotheses_01)
        dp_l22 = devpav_new(lrs, all_hypotheses_01)

        # Determine metrics for too extreme data
        lrs = np.concatenate((SS_e1, DS_e1))

        dp_e1 = devpav(lrs, all_hypotheses_01)
        dp_e11 = scaled_devpav(lrs, all_hypotheses_01)
        dp_e12 = devpav_new(lrs, all_hypotheses_01)

        lrs = np.concatenate((SS_e2, DS_e2))

        dp_e2 = devpav(lrs, all_hypotheses_01)
        dp_e21 = scaled_devpav(lrs, all_hypotheses_01)
        dp_e22 = devpav_new(lrs, all_hypotheses_01)

        # Determine metrics for too weak data
        lrs = np.concatenate((SS_w1, DS_w1))

        dp_w1 = devpav(lrs, all_hypotheses_01)
        dp_w11 = scaled_devpav(lrs, all_hypotheses_01)
        dp_w12 = devpav_new(lrs, all_hypotheses_01)

        lrs = np.concatenate((SS_w2, DS_w2))

        dp_w2 = devpav(lrs, all_hypotheses_01)
        dp_w21 = scaled_devpav(lrs, all_hypotheses_01)
        dp_w22 = devpav_new(lrs, all_hypotheses_01)

        # Append metric values to arrays
        dev_p.append(dp_p)
        sdev_p.append(dp_p1)
        cdev_p.append(dp_p2)

        dev_r1.append(dp_r1)
        sdev_r1.append(dp_r11)
        cdev_r1.append(dp_r12)

        dev_r2.append(dp_r2)
        sdev_r2.append(dp_r21)
        cdev_r2.append(dp_r22)

        dev_l1.append(dp_l1)
        sdev_l1.append(dp_l11)
        cdev_l1.append(dp_l12)

        dev_l2.append(dp_l2)
        sdev_l2.append(dp_l21)
        cdev_l2.append(dp_l22)

        dev_e1.append(dp_e1)
        sdev_e1.append(dp_e11)
        cdev_e1.append(dp_e12)

        dev_e2.append(dp_e2)
        sdev_e2.append(dp_e21)
        cdev_e2.append(dp_e22)

        dev_w1.append(dp_w1)
        sdev_w1.append(dp_w11)
        cdev_w1.append(dp_w12)

        dev_w2.append(dp_w2)
        sdev_w2.append(dp_w21)
        cdev_w2.append(dp_w22)

    except Exception as e:
        print(f"An error occurred in iteration {i}: {e}")
        continue

# Collect results of normal devPAV in dictionary
results_dev = {
    'Perfect': dev_p,
    'Right c=1': dev_r1,
    'Left c=1': dev_l1,
    'Extreme c=1.5': dev_e1,
    'Weak c=1.5': dev_w1,
    'Right c=2': dev_r2,
    'Left c=2': dev_l2,
    'Extreme c=2.5': dev_e2,
    'Weak c=2.5': dev_w2,
}
df_results = pd.DataFrame(results_dev)

# Compute the overlap percentage
devpavs = np.array(list(results_dev.values()))
overlap_devs = overlap(devpavs)
print('Devpavs overlap:', overlap_devs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 2)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Normal devPAV', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of scaled devPAV in dictionary
results_sdev = {
    'Perfect': sdev_p,
    'Right c=1': sdev_r1,
    'Left c=1': sdev_l1,
    'Extreme c=1.5': sdev_e1,
    'Weak c=1.5': sdev_w1,
    'Right c=2': sdev_r2,
    'Left c=2': sdev_l2,
    'Extreme c=2.5': sdev_e2,
    'Weak c=2.5': sdev_w2,
}
df_results = pd.DataFrame(results_sdev)

# Compute overlap percentage
scaled_devpavs = np.array(list(results_sdev.values()))
overlap_scaledevs = overlap(scaled_devpavs)
print('Scaled devpavs overlap:', overlap_scaledevs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Scaled devPAV', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of smoothed devPAV in dictionary
results_cdev = {
    'Perfect': cdev_p,
    'Right c=1': cdev_r1,
    'Left c=1': cdev_l1,
    'Extreme c=1.5': cdev_e1,
    'Weak c=1.5': cdev_w1,
    'Right c=2': cdev_r2,
    'Left c=2': cdev_l2,
    'Extreme c=2.5': cdev_e2,
    'Weak c=2.5': cdev_w2,
}
df_results = pd.DataFrame(results_cdev)

# Compute overlap percentage
smooth_devpavs = np.array(list(results_cdev.values()))
overlap_smoothdevs = overlap(smooth_devpavs)
print('Smoothed devpavs overlap:', overlap_smoothdevs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Smoothed devPAV', fontsize=16)
plt.tight_layout()
plt.show()