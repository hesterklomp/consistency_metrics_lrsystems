from functions_thesis import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Set working directory (needs to be updated to the appropriate directory)
os.chdir('C:/Users/heste/OneDrive/Documents/TU Delft/THESIS/Code/code hannig/code')

# Initialize mean, variation, amount of data and scaling factors
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

# Initialize arrays to store fid values using average of medians
fid_p = []
fid_r1 = []
fid_r2 = []
fid_l1 = []
fid_l2 = []
fid_e1 = []
fid_e2 = []
fid_w1 = []
fid_w2 = []

# Initialize arrays to store fid values using scaled average of medians
mfid_p = []
mfid_r1 = []
mfid_r2 = []
mfid_l1 = []
mfid_l2 = []
mfid_e1 = []
mfid_e2 = []
mfid_w1 = []
mfid_w2 = []

# Initialize arrays to store fid values using zero-one score
zfid_p = []
zfid_r1 = []
zfid_r2 = []
zfid_l1 = []
zfid_l2 = []
zfid_e1 = []
zfid_e2 = []
zfid_w1 = []
zfid_w2 = []

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
        calibration = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_p, DS_p))),
                                          'labels': ['P'] * n_ss + ['D'] * n_sd}),
                            nfid=100, AUC=True)
        calibration_r = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_r1, DS_r1))),
                                               'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                 nfid=100, AUC=True)

        calibration_r2 = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_r2, DS_r2))),
                                                'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                  nfid=100, AUC=True)

        calibration_l = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_l1, DS_l1))),
                                               'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                 nfid=100, AUC=True)

        calibration_l2 = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_l2, DS_l2))),
                                                'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                  nfid=100, AUC=True)

        calibration_e = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_e1, DS_e1))),
                                                'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                  nfid=100, AUC=True)

        calibration_e2 = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_e2, DS_e2))),
                                                'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                  nfid=100, AUC=True)

        calibration_w = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_w1, DS_w1))),
                                               'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                 nfid=100, AUC=True)

        calibration_w2 = LRtestNP(pd.DataFrame({'LLR': np.log10(np.concatenate((SS_w2, DS_w2))),
                                                'labels': ['P'] * n_ss + ['D'] * n_sd}),
                                  nfid=100, AUC=True)

        # Append metric values to arrays
        fid_p.append(calibration['calib'][0])
        mfid_p.append(calibration['calib2'][0])
        zfid_p.append(calibration['calib3'])

        fid_r1.append(calibration_r['calib'][0])
        mfid_r1.append(calibration_r['calib2'][0])
        zfid_r1.append(calibration_r['calib3'])

        fid_r2.append(calibration_r2['calib'][0])
        mfid_r2.append(calibration_r2['calib2'][0])
        zfid_r2.append(calibration_r2['calib3'])

        fid_l1.append(calibration_l['calib'][0])
        mfid_l1.append(calibration_l['calib2'][0])
        zfid_l1.append(calibration_l['calib3'])

        fid_l2.append(calibration_l2['calib'][0])
        mfid_l2.append(calibration_l2['calib2'][0])
        zfid_l2.append(calibration_l2['calib3'])

        fid_e1.append(calibration_e['calib'][0])
        mfid_e1.append(calibration_e['calib2'][0])
        zfid_e1.append(calibration_e['calib3'])

        fid_e2.append(calibration_e2['calib'][0])
        mfid_e2.append(calibration_e2['calib2'][0])
        zfid_e2.append(calibration_e2['calib3'])

        fid_w1.append(calibration_w['calib'][0])
        mfid_w1.append(calibration_w['calib2'][0])
        zfid_w1.append(calibration_w['calib3'])

        fid_w2.append(calibration_w2['calib'][0])
        mfid_w2.append(calibration_w2['calib2'][0])
        zfid_w2.append(calibration_w2['calib3'])

    except Exception as e:
        print(f"An error occurred in iteration {i}: {e}")
        continue

# Collect results of Fid using average of medians in dictionary
results_fid = {
    'Perfect': fid_p,
    'Right c=1': fid_r1,
    'Left c=1': fid_l1,
    'Extreme c=1.5': fid_e1,
    'Weak c=1.5': fid_w1,
    'Right c=2': fid_r2,
    'Left c=2': fid_l2,
    'Extreme c=2.5': fid_e2,
    'Weak c=2.5': fid_w2,
}
df_results = pd.DataFrame(results_fid)

# Compute overlap percentage
fids_1 = np.array(list(results_fid.values()))
overlap_1 = overlap(fids_1)
print('Median fid overlap:', overlap_1)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Fiducial metric', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of Fid using scaled average of medians in dictionary
results_fid2 = {
    'Perfect': mfid_p,
    'Right c=1': mfid_r1,
    'Left c=1': mfid_l1,
    'Extreme c=1.5': mfid_e1,
    'Weak c=2.5': mfid_w2,
    'Right c=2': mfid_r2,
    'Left c=2': mfid_l2,
    'Extreme c=2.5': mfid_e2,
    'Weak c=1.5': mfid_w1,
}
df_results2 = pd.DataFrame(results_fid2)

# Compute overlap percentage
fids_2 = np.array(list(results_fid2.values()))
overlap_2 = overlap(fids_2)
print('Scaled fid overlap:', overlap_2)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results2.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results2.columns):
    sns.violinplot(data=df_results2[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 2.0)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Fiducial metric, scaled', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of zero-one fid metric in dictionary
results_fid3 = {
    'Perfect': zfid_p,
    'Right c=1': zfid_r1,
    'Left c=1': zfid_l1,
    'Extreme c=1.5': zfid_e1,
    'Weak c=1.5': zfid_w1,
    'Right c=2': zfid_r2,
    'Left c=2': zfid_l2,
    'Extreme c=2.5': zfid_e2,
    'Weak c=2.5': zfid_w2,
}
df_results3 = pd.DataFrame(results_fid3)

# Compute overlap percentage
fids_3 = np.array(list(results_fid3.values()))
overlap_3 = overlap(fids_3)
print('Zero-one fids overlap:', overlap_3)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results3.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results3.columns):
    sns.violinplot(data=df_results3[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Fiducial metric, zero-one', fontsize=16)
plt.tight_layout()
plt.show()