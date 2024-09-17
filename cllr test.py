import numpy as np
import matplotlib.pyplot as plt
import math
from math import comb, sqrt
from functions_thesis import *
import pandas as pd
from lir import *
import seaborn as sns

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

# Initialize arrays to store normal Cllr values
cllr_p = []
cllr_r1 = []
cllr_r2 = []
cllr_l1 = []
cllr_l2 = []
cllr_e1 = []
cllr_e2 = []
cllr_w1 = []
cllr_w2 = []

# Initialize arrays to store Cllr values using the Brier score
brier_p = []
brier_r1 = []
brier_r2 = []
brier_l1 = []
brier_l2 = []
brier_e1 = []
brier_e2 = []
brier_w1 = []
brier_w2 = []

# Initialize arrays to store Cllr values using the zero-one score
zerone_p = []
zerone_r1 = []
zerone_r2 = []
zerone_l1 = []
zerone_l2 = []
zerone_e1 = []
zerone_e2 = []
zerone_w1 = []
zerone_w2 = []

# Initialize arrays to store Cllr values using the spherical scoring rule
spher_p = []
spher_r1 = []
spher_r2 = []
spher_l1 = []
spher_l2 = []
spher_e1 = []
spher_e2 = []
spher_w1 = []
spher_w2 = []

# Create an array of log 10 prior odds ranging from -5 to 5
log_prior_odds = np.linspace(-5, 5, num=100)  # Adjust the number of points as needed
# Calculate prior odds array
prior_odds = 10 ** log_prior_odds

for i in range(N):
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
    all_data = np.concatenate((SS_p, DS_p))
    all_hypotheses = np.concatenate((np.array(['H1'] * len(SS_p)), np.array(['H2'] * len(DS_p))))
    all_hypotheses_01 = np.where(all_hypotheses == 'H1', 1, 0)

    # PAV lrs
    cal = lir.IsotonicCalibrator()
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Determine normal cllr
    cllr1 = cllr(all_data, all_hypotheses_01)
    cllr_p.append(cllr1)

    # Determine Brier-score difference between normal and PAV data
    brierp_1 = brier(all_data, all_hypotheses_01)
    brierp_2 = brier(lrmin, all_hypotheses_01)
    brierp = brierp_1 - brierp_2
    brier_p.append(brierp)

    # Determine zero-one cllr
    zeronep_1 = zero_one(all_data, all_hypotheses_01)
    zeronep_2 = zero_one(lrmin, all_hypotheses_01)
    zeronep = zeronep_1 - zeronep_2
    zerone_p.append(zeronep)

    # Determine spherical cllr
    spherp_1 = spherical(all_data, all_hypotheses_01)
    spherp_2 = spherical(lrmin, all_hypotheses_01)
    spherp = spherp_1 - spherp_2
    spher_p.append(spherp)

    # Determine metrics for data skewed to the right
    all_data = np.concatenate((SS_r1, DS_r1))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr2 = cllr(all_data, all_hypotheses_01)
    cllr_r1.append(cllr2)

    # Brier cllr
    brierr1_1 = brier(all_data, all_hypotheses_01)
    brierr1_2 = brier(lrmin, all_hypotheses_01)
    brierr1 = brierr1_1 - brierr1_2
    brier_r1.append(brierr1)

    # Zero-one cllr
    zeroner1_1 = zero_one(all_data, all_hypotheses_01)
    zeroner1_2 = zero_one(lrmin, all_hypotheses_01)
    zeroner1 = zeroner1_1 - zeroner1_2
    zerone_r1.append(zeroner1)

    # Spherical cllr
    spherr1_1 = spherical(all_data, all_hypotheses_01)
    spherr1_2 = spherical(lrmin, all_hypotheses_01)
    spherr1 = spherr1_1 - spherr1_2
    spher_r1.append(spherr1)

    all_data = np.concatenate((SS_r2, DS_r2))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr3 = cllr(all_data, all_hypotheses_01)
    cllr_r2.append(cllr3)

    # Brier cllr
    brierr2_1 = brier(all_data, all_hypotheses_01)
    brierr2_2 = brier(lrmin, all_hypotheses_01)
    brierr2 = brierr2_1 - brierr2_2
    brier_r2.append(brierr2)

    # Zero-one cllr
    zeroner2_1 = zero_one(all_data, all_hypotheses_01)
    zeroner2_2 = zero_one(lrmin, all_hypotheses_01)
    zeroner2 = zeroner2_1 - zeroner2_2
    zerone_r2.append(zeroner2)

    # Spherical cllr
    spherr2_1 = spherical(all_data, all_hypotheses_01)
    spherr2_2 = spherical(lrmin, all_hypotheses_01)
    spherr2 = spherr2_1 - spherr2_2
    spher_r2.append(spherr2)

    # Determine metrics for data skewed to the left
    all_data = np.concatenate((SS_l1, DS_l1))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr4 = cllr(all_data, all_hypotheses_01)
    cllr_l1.append(cllr4)

    # Brier cllr
    brierl1_1 = brier(all_data, all_hypotheses_01)
    brierl1_2 = brier(lrmin, all_hypotheses_01)
    brierl1 = brierl1_1 - brierl1_2
    brier_l1.append(brierl1)

    # Zero-one cllr
    zeronel1_1 = zero_one(all_data, all_hypotheses_01)
    zeronel1_2 = zero_one(lrmin, all_hypotheses_01)
    zeronel1 = zeronel1_1 - zeronel1_2
    zerone_l1.append(zeronel1)

    # Spherical cllr
    spherl1_1 = spherical(all_data, all_hypotheses_01)
    spherl1_2 = spherical(lrmin, all_hypotheses_01)
    spherl1 = spherl1_1 - spherl1_2
    spher_l1.append(spherl1)

    all_data = np.concatenate((SS_l2, DS_l2))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr5 = cllr(all_data, all_hypotheses_01)
    cllr_l2.append(cllr5)

    # Brier cllr
    brierl2_1 = brier(all_data, all_hypotheses_01)
    brierl2_2 = brier(lrmin, all_hypotheses_01)
    brierl2 = brierl2_1 - brierl2_2
    brier_l2.append(brierl2)

    # Zero-one cllr
    zeronel2_1 = zero_one(all_data, all_hypotheses_01)
    zeronel2_2 = zero_one(lrmin, all_hypotheses_01)
    zeronel2 = zeronel2_1 - zeronel2_2
    zerone_l2.append(zeronel2)

    # Spherical cllr
    spherl2_1 = spherical(all_data, all_hypotheses_01)
    spherl2_2 = spherical(lrmin, all_hypotheses_01)
    spherl2 = spherl2_1 - spherl2_2
    spher_l2.append(spherl2)

    # Determine metrics for too extreme data
    all_data = np.concatenate((SS_e1, DS_e1))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr6 = cllr(all_data, all_hypotheses_01)
    cllr_e1.append(cllr6)

    # Brier cllr
    briere1_1 = brier(all_data, all_hypotheses_01)
    briere1_2 = brier(lrmin, all_hypotheses_01)
    briere1 = briere1_1 - briere1_2
    brier_e1.append(briere1)

    # Zero-one cllr
    zeronee1_1 = zero_one(all_data, all_hypotheses_01)
    zeronee1_2 = zero_one(lrmin, all_hypotheses_01)
    zeronee1 = zeronee1_1 - zeronee1_2
    zerone_e1.append(zeronee1)

    # Spherical cllr
    sphere1_1 = spherical(all_data, all_hypotheses_01)
    sphere1_2 = spherical(lrmin, all_hypotheses_01)
    sphere1 = sphere1_1 - sphere1_2
    spher_e1.append(sphere1)

    all_data = np.concatenate((SS_e2, DS_e2))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr7 = cllr(all_data, all_hypotheses_01)
    cllr_e2.append(cllr7)

    # Brier cllr
    briere2_1 = brier(all_data, all_hypotheses_01)
    briere2_2 = brier(lrmin, all_hypotheses_01)
    briere2 = briere2_1 - briere2_2
    brier_e2.append(briere2)

    # Zero-one cllr
    zeronee2_1 = zero_one(all_data, all_hypotheses_01)
    zeronee2_2 = zero_one(lrmin, all_hypotheses_01)
    zeronee2 = zeronee2_1 - zeronee2_2
    zerone_e2.append(zeronee2)

    # Spherical cllr
    sphere2_1 = spherical(all_data, all_hypotheses_01)
    sphere2_2 = spherical(lrmin, all_hypotheses_01)
    sphere2 = sphere2_1 - sphere2_2
    spher_e2.append(sphere2)

    # Determine metrics for too weak data
    all_data = np.concatenate((SS_w1, DS_w1))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr8 = cllr(all_data, all_hypotheses_01)
    cllr_w1.append(cllr8)

    # Brier cllr
    brierw1_1 = brier(all_data, all_hypotheses_01)
    brierw1_2 = brier(lrmin, all_hypotheses_01)
    brierw1 = brierw1_1 - brierw1_2
    brier_w1.append(brierw1)

    # Zero-one cllr
    zeronew1_1 = zero_one(all_data, all_hypotheses_01)
    zeronew1_2 = zero_one(lrmin, all_hypotheses_01)
    zeronew1 = zeronew1_1 - zeronew1_2
    zerone_w1.append(zeronew1)

    # Spherical cllr
    spherw1_1 = spherical(all_data, all_hypotheses_01)
    spherw1_2 = spherical(lrmin, all_hypotheses_01)
    spherw1 = spherw1_1 - spherw1_2
    spher_w1.append(spherw1)

    all_data = np.concatenate((SS_w2, DS_w2))
    lrmin = cal.fit_transform(to_probability(all_data), all_hypotheses_01)

    # Normal cllr
    cllr9 = cllr(all_data, all_hypotheses_01)
    cllr_w2.append(cllr9)

    # Brier cllr
    brierw2_1 = brier(all_data, all_hypotheses_01)
    brierw2_2 = brier(lrmin, all_hypotheses_01)
    brierw2 = brierw2_1 - brierw2_2
    brier_w2.append(brierw2)

    # Zero-one cllr
    zeronew2_1 = zero_one(all_data, all_hypotheses_01)
    zeronew2_2 = zero_one(lrmin, all_hypotheses_01)
    zeronew2 = zeronew2_1 - zeronew2_2
    zerone_w2.append(zeronew2)

    # Spherical cllr
    spherw2_1 = spherical(all_data, all_hypotheses_01)
    spherw2_2 = spherical(lrmin, all_hypotheses_01)
    spherw2 = spherw2_1 - spherw2_2
    spher_w2.append(spherw2)

# Collect results of normal Cllr in dictionary
results_cllr = {
    'Perfect': cllr_p,
    'Right c=1': cllr_r1,
    'Left c=1': cllr_l1,
    'Extreme c=1.5': cllr_e1,
    'Weak c=1.5': cllr_w1,
    'Right c=2': cllr_r2,
    'Left c=2': cllr_l2,
    'Extreme c=2.5': cllr_e2,
    'Weak c=2.5': cllr_w2,
}
df_results = pd.DataFrame(results_cllr)

# Compute overlap percentage
cllr_normals = np.array(list(results_cllr.values()))
overlap_normals = overlap(cllr_normals)
print('Normals overlap:', overlap_normals)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Cllr normal', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of Brier Cllr in dictionary
results_brier = {
    'Perfect': brier_p,
    'Right c=1': brier_r1,
    'Left c=1': brier_l1,
    'Extreme c=1.5': brier_e1,
    'Weak c=1.5': brier_w1,
    'Right c=2': brier_r2,
    'Left c=2': brier_l2,
    'Extreme c=2.5': brier_e2,
    'Weak c=2.5': brier_w2,
}
df_resultsb = pd.DataFrame(results_brier)

# Compute overlap percentage
cllr_briers = np.array(list(results_brier.values()))
overlap_briers = overlap(cllr_briers)
print('Briers overlap:', overlap_briers)

# Plot results
fig, axes = plt.subplots(ncols=len(df_resultsb.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_resultsb.columns):
    sns.violinplot(data=df_resultsb[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0.0,0.05)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Brier', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of zero-one Cllr in dictionary
results_zerone = {
    'Perfect': zerone_p,
    'Right c=1': zerone_r1,
    'Left c=1': zerone_l1,
    'Extreme c=1.5': zerone_e1,
    'Weak c=1.5': zerone_w1,
    'Right c=2': zerone_r2,
    'Left c=2': zerone_l2,
    'Extreme c=2.5': zerone_e2,
    'Weak c=2.5': zerone_w2,
}
df_resultszer = pd.DataFrame(results_zerone)

# Compute overlap percentage
cllr_zerons = np.array(list(results_zerone.values()))
overlap_zerons = overlap(cllr_zerons)
print('Zero ones overlap:', overlap_zerons)

# Plot results
fig, axes = plt.subplots(ncols=len(df_resultszer.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_resultszer.columns):
    sns.violinplot(data=df_resultszer[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylabel('')  # Remove y-axis label
plt.suptitle('Zero-one', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of spherical Cllr in dictionary
results_spher = {
    'Perfect': spher_p,
    'Right c=1': spher_r1,
    'Left c=1': spher_l1,
    'Extreme c=1.5': spher_e1,
    'Weak c=1.5': spher_w1,
    'Right c=2': spher_r2,
    'Left c=2': spher_l2,
    'Extreme c=2.5': spher_e2,
    'Weak c=2.5': spher_w2,
}
df_resultssph = pd.DataFrame(results_spher)

# Compute overlap percentage
cllr_sphers = np.array(list(results_spher.values()))
overlap_sphers = overlap(cllr_sphers)
print('Sphericals overlap:', overlap_sphers)

# Plot results
fig, axes = plt.subplots(ncols=len(df_resultssph.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_resultssph.columns):
    sns.violinplot(data=df_resultssph[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylabel('')
plt.suptitle('Spherical', fontsize=16)
plt.tight_layout()
plt.show()