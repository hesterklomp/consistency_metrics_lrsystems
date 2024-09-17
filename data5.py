import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from functions_thesis import *

# Initialize constants
c_1 = 1
c_2 = 2
d_1 = 1.5
d_2 = 2.5
N = 1000
n_ss = 150
n_ds = 450

# Initialize lists to store metric values
dev_p = []
dev_r1 = []
dev_r2 = []
dev_l1 = []
dev_l2 = []
dev_e1 = []
dev_e2 = []
dev_w1 = []
dev_w2 = []

cllr_p = []
cllr_r1 = []
cllr_r2 = []
cllr_l1 = []
cllr_l2 = []
cllr_e1 = []
cllr_e2 = []
cllr_w1 = []
cllr_w2 = []

fid_p = []
fid_r1 = []
fid_r2 = []
fid_l1 = []
fid_l2 = []
fid_e1 = []
fid_e2 = []
fid_w1 = []
fid_w2 = []

# Unpack data
log_data_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data + Code\\data_thesis\\data5\\LLR_KM.csv', newline='') as csvfile:
    # Create a CSV reader object
    csv_reader = csv.reader(csvfile)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        log_data_SS.append(float(row[0]))

log_data_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data + Code\\data_thesis\\data5\\LLR_KNM.csv', newline='') as csvfile:
    # Create a CSV reader object
    csv_reader = csv.reader(csvfile)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        log_data_DS.append(float(row[0]))

# Plot original data
plt.hist(log_data_SS, bins=15, color='blue', density=True, alpha=0.5, label='SS')
plt.hist(log_data_DS, bins=15, color='red', density=True, alpha=0.5, label='DS')
plt.xlabel('LLR')
plt.ylabel('Relative frequency')
plt.legend()
plt.title('Original data')
plt.show()

# Print number of values
print('Number of SS values:', len(log_data_SS))
print('Number of DS values:', len(log_data_DS))

# Calculate percentiles for the SS dataset
percentiles_SS = np.percentile(log_data_SS, [5, 50, 95])
print(f"Percentiles for SS dataset: 5th: {percentiles_SS[0]}, 50th: {percentiles_SS[1]}, 95th: {percentiles_SS[2]}")

# Calculate percentiles for the DS dataset
percentiles_DS = np.percentile(log_data_DS, [5, 50, 95])
print(f"Percentiles for DS dataset: 5th: {percentiles_DS[0]}, 50th: {percentiles_DS[1]}, 95th: {percentiles_DS[2]}")

# Calculate LSS and LDS densities for consistent LR-system
frequencies = frequency_creator(log_data_DS)
LDS_frequencies = frequencies[0]
LSS_frequencies = frequencies[1]
stretched_LLRs = frequencies[2]

for i in range(N):
    print(i)

    # Consistent data
    samples_LDS = np.random.choice(stretched_LLRs, size=n_ds, p=LDS_frequencies)
    samples_LSS = np.random.choice(stretched_LLRs, size=n_ss, p=LSS_frequencies)

    SS_p = np.power(10, samples_LSS)
    DS_p = np.power(10, samples_LDS)

    # Calculate metrics
    metrics_p = calculate_metrics_all(SS_p, DS_p)
    dp_p = metrics_p[1]
    c_p = metrics_p[0]
    cal_p = metrics_p[2]

    # Data skewed to right
    LSS_r1 = samples_LSS + c_1
    LDS_r1 = samples_LDS + c_1
    SS_r1 = np.power(10, LSS_r1)
    DS_r1 = np.power(10, LDS_r1)

    metrics_r1 = calculate_metrics_all(SS_r1, DS_r1)
    dp_r1 = metrics_r1[1]
    c_r1 = metrics_r1[0]
    cal_r1 = metrics_r1[2]

    LSS_r2 = samples_LSS + c_2
    LDS_r2 = samples_LDS + c_2
    SS_r2 = np.power(10, LSS_r2)
    DS_r2 = np.power(10, LDS_r2)

    metrics_r2 = calculate_metrics_all(SS_r2, DS_r2)
    dp_r2 = metrics_r2[1]
    c_r2 = metrics_r2[0]
    cal_r2 = metrics_r2[2]

    # Data skewed to left
    LSS_l1 = samples_LSS - c_1
    LDS_l1 = samples_LDS - c_1
    SS_l1 = np.power(10, LSS_l1)
    DS_l1 = np.power(10, LDS_l1)

    metrics_l1 = calculate_metrics_all(SS_l1, DS_l1)
    dp_l1 = metrics_l1[1]
    c_l1 = metrics_l1[0]
    cal_l1 = metrics_l1[2]

    LSS_l2 = samples_LSS - c_2
    LDS_l2 = samples_LDS - c_2
    SS_l2 = np.power(10, LSS_l2)
    DS_l2 = np.power(10, LDS_l2)

    metrics_l2 = calculate_metrics_all(SS_l2, DS_l2)
    dp_l2 = metrics_l2[1]
    c_l2 = metrics_l2[0]
    cal_l2 = metrics_l2[2]

    # Data extreme
    LSS_e1 = d_1 * samples_LSS
    LDS_e1 = d_1 * samples_LDS
    SS_e1 = np.power(10, LSS_e1)
    DS_e1 = np.power(10, LDS_e1)

    metrics_e1 = calculate_metrics_all(SS_e1, DS_e1)
    dp_e1 = metrics_e1[1]
    c_e1 = metrics_e1[0]
    cal_e1 = metrics_e1[2]

    LSS_e2 = d_2 * samples_LSS
    LDS_e2 = d_2 * samples_LDS
    SS_e2 = np.power(10, LSS_e2)
    DS_e2 = np.power(10, LDS_e2)

    metrics_e2 = calculate_metrics_all(SS_e2, DS_e2)
    dp_e2 = metrics_e2[1]
    c_e2 = metrics_e2[0]
    cal_e2 = metrics_e2[2]

    # Data weak
    LSS_w1 = (1 / d_1) * samples_LSS
    LDS_w1 = (1 / d_1) * samples_LDS
    SS_w1 = np.power(10, LSS_w1)
    DS_w1 = np.power(10, LDS_w1)

    metrics_w1 = calculate_metrics_all(SS_w1, DS_w1)
    dp_w1 = metrics_w1[1]
    c_w1 = metrics_w1[0]
    cal_w1 = metrics_w1[2]

    LSS_w2 = (1 / d_2) * samples_LSS
    LDS_w2 = (1 / d_2) * samples_LDS
    SS_w2 = np.power(10, LSS_w2)
    DS_w2 = np.power(10, LDS_w2)

    metrics_w2 = calculate_metrics_all(SS_w2, DS_w2)
    dp_w2 = metrics_w2[1]
    c_w2 = metrics_w2[0]
    cal_w2 = metrics_w2[2]

    # Store metric values
    dev_p.append(dp_p)
    dev_r1.append(dp_r1)
    dev_r2.append(dp_r2)
    dev_l1.append(dp_l1)
    dev_l2.append(dp_l2)
    dev_e1.append(dp_e1)
    dev_e2.append(dp_e2)
    dev_w1.append(dp_w1)
    dev_w2.append(dp_w2)

    cllr_p.append(c_p)
    cllr_r1.append(c_r1)
    cllr_r2.append(c_r2)
    cllr_l1.append(c_l1)
    cllr_l2.append(c_l2)
    cllr_e1.append(c_e1)
    cllr_e2.append(c_e2)
    cllr_w1.append(c_w1)
    cllr_w2.append(c_w2)

    fid_p.append(cal_p)
    fid_r1.append(cal_r1)
    fid_r2.append(cal_r2)
    fid_l1.append(cal_l1)
    fid_l2.append(cal_l2)
    fid_e1.append(cal_e1)
    fid_e2.append(cal_e2)
    fid_w1.append(cal_w1)
    fid_w2.append(cal_w2)

# Collect results dev
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

# Calculate overlap percentage
devs = np.array(list(results_dev.values()))
overlap_devs = overlap(devs)
print('Devpav overlap:', overlap_devs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 1)
plt.suptitle('devPAV', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results cllr
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

# Calculate overlap percentage
cllrs = np.array(list(results_cllr.values()))
overlap_cllrs = overlap(cllrs)
print('Cllr overlap:', overlap_cllrs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 1)
plt.suptitle('Cllr', fontsize=16)
plt.tight_layout()
plt.show()

# Collect Fid results
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

# Calculate overlap percentage
fids = np.array(list(results_fid.values()))
overlap_fids = overlap(fids)
print('Fid overlap:', overlap_fids)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_ylim(0, 1)
plt.suptitle('Fid', fontsize=16)
plt.tight_layout()
plt.show()
