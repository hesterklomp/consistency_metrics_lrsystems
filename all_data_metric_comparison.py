import numpy as np
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from functions_thesis import *

# Initialize values
N = 10
MU_ss = 6
SIGMA = sqrt(2 * MU_ss)
n_ss_small = 50
n_ds_small = 150
n_ss_medium = 150
n_ds_medium = 450
n_ss_large = 300
n_ds_large = 900

# Initialize arrays to store metrics values
crossval_ELUB_small_dev = []
crossval_ELUB_small_cllr = []

crossval_ELUB_medium_dev = []
crossval_ELUB_medium_cllr = []
crossval_ELUB_medium_fid = []

crossval_ELUB_large_dev = []
crossval_ELUB_large_cllr = []
crossval_ELUB_large_fid = []

crossval_noELUB_small_dev = []
crossval_noELUB_small_cllr = []

crossval_noELUB_medium_dev = []
crossval_noELUB_medium_cllr = []
crossval_noELUB_medium_fid = []

crossval_noELUB_large_dev = []
crossval_noELUB_large_cllr = []
crossval_noELUB_large_fid = []

data_fig29_small_dev = []
data_fig29_small_cllr = []

data_fig29_medium_dev = []
data_fig29_medium_cllr = []
data_fig29_medium_fid = []

data_fig29_large_dev = []
data_fig29_large_cllr = []
data_fig29_large_fid = []

final_noELUB_small_dev = []
final_noELUB_small_cllr = []

final_noELUB_medium_dev = []
final_noELUB_medium_cllr = []
final_noELUB_medium_fid = []

final_noELUB_large_dev = []
final_noELUB_large_cllr = []
final_noELUB_large_fid = []

final_ELUB_small_dev = []
final_ELUB_small_cllr = []

final_ELUB_medium_dev = []
final_ELUB_medium_cllr = []
final_ELUB_medium_fid = []

final_ELUB_large_dev = []
final_ELUB_large_cllr = []
final_ELUB_large_fid = []

hulzen_small_dev = []
hulzen_small_cllr = []

hulzen_medium_dev = []
hulzen_medium_cllr = []
hulzen_medium_fid = []

hulzen_large_dev = []
hulzen_large_cllr = []
hulzen_large_fid = []

benzine_small_dev = []
benzine_small_cllr = []

benzine_medium_dev = []
benzine_medium_cllr = []
benzine_medium_fid = []

benzine_large_dev = []
benzine_large_cllr = []
benzine_large_fid = []

normal_small_dev = []
normal_small_cllr = []

normal_medium_dev = []
normal_medium_cllr = []
normal_medium_fid = []

normal_large_dev = []
normal_large_cllr = []
normal_large_fid = []

# Load all the datasets
crossval_ELUB_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\crossval_ELUB\\LLR_KM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        crossval_ELUB_SS.append(float(row[0]))

crossval_ELUB_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\crossval_ELUB\\LLR_KNM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        crossval_ELUB_DS.append(float(row[0]))

data_fig29_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig29_M2\\LLR_KM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        data_fig29_SS.append(float(row[0]))

data_fig29_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig29_M2\\LLR_KNM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        data_fig29_DS.append(float(row[0]))

final_noELUB_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\finalmodel_noELUB\\LLR_KM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        final_noELUB_SS.append(float(row[0]))

final_noELUB_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\finalmodel_noELUB\\LLR_KNM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        final_noELUB_DS.append(float(row[0]))

final_ELUB_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\finalmodel_ELUB\\LLR_KM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        final_ELUB_SS.append(float(row[0]))

final_ELUB_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\finalmodel_ELUB\\LLR_KNM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        final_ELUB_DS.append(float(row[0]))

crossval_noELUB_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\crossval_noELUB\\LLR_KM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        crossval_noELUB_SS.append(float(row[0]))

crossval_noELUB_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Scratch-artikel\\Fig9_M2\\crossval_noELUB\\LLR_KNM.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        crossval_noELUB_DS.append(float(row[0]))

hulzen_SS = []
hulzen_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Peter\\GSR\\data_timo\\lrs_huls_paren.csv',
          newline='') as csvfile:
    # Create a CSV reader object
    csv_reader = csv.DictReader(csvfile)

    # Skip the header row (optional, depending on the structure of the CSV)
    next(csv_reader)

    for row in csv_reader:
            lr_value = row['lr']
            label = row['label']  # Assuming 'label' is the column name

            # Check for the label and append the data to the corresponding list
            if label == '1.0':
                if lr_value.lower() == 'inf':
                    hulzen_SS.append(float('inf'))
                else:
                    hulzen_SS.append(math.log10(float(lr_value)))
            elif label == '0.0':
                if lr_value.lower() == 'inf':
                    hulzen_DS.append(float('inf'))
                else:
                    hulzen_DS.append(math.log10(float(lr_value)))


benzine_SS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Peter\\benzines\\10LLRssame.csv',
          newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)

    for row in csv_reader:
        benzine_SS.append(float(row[1]))


benzine_DS = []
with open('C:\\Users\\heste\\OneDrive\\Documents\\TU Delft\\THESIS\\Data\\Peter\\benzines\\10LLRsdif.csv',
          newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)

    for row in csv_reader:
        benzine_DS.append(float(row[1]))


# For each dataset, create a consistent LR-system
cv_noELUB_frequencies = frequency_creator(crossval_noELUB_DS)
cv_noELUB_LDS_frequencies = cv_noELUB_frequencies[0]
cv_noELUB_LSS_frequencies = cv_noELUB_frequencies[1]
cv_noELUB_stretched_LLRs = cv_noELUB_frequencies[2]

cv_ELUB_frequencies = frequency_creator(crossval_ELUB_DS)
cv_ELUB_LDS_frequencies = cv_ELUB_frequencies[0]
cv_ELUB_LSS_frequencies = cv_ELUB_frequencies[1]
cv_ELUB_stretched_LLRs = cv_ELUB_frequencies[2]

final_noELUB_frequencies = frequency_creator(final_noELUB_DS)
final_noELUB_LDS_frequencies = final_noELUB_frequencies[0]
final_noELUB_LSS_frequencies = final_noELUB_frequencies[1]
final_noELUB_stretched_LLRs = final_noELUB_frequencies[2]

final_ELUB_frequencies = frequency_creator(final_ELUB_DS)
final_ELUB_LDS_frequencies = final_ELUB_frequencies[0]
final_ELUB_LSS_frequencies = final_ELUB_frequencies[1]
final_ELUB_stretched_LLRs = final_ELUB_frequencies[2]

data_fig29_frequencies = frequency_creator(data_fig29_DS)
fig29_LDS_frequencies = data_fig29_frequencies[0]
fig29_LSS_frequencies = data_fig29_frequencies[1]
fig29_stretched_LLRs = data_fig29_frequencies[2]

hulzen_frequencies = frequency_creator(hulzen_DS)
hulzen_LDS_frequencies = hulzen_frequencies[0]
hulzen_LSS_frequencies = hulzen_frequencies[1]
hulzen_stretched_LLRs = hulzen_frequencies[2]


benzine_frequencies = frequency_creator(benzine_DS)
benzine_LDS_frequencies = benzine_frequencies[0]
benzine_LSS_frequencies = benzine_frequencies[1]
benzine_stretched_LLRs = benzine_frequencies[2]


# Loop to determine the metrics
for i in range(N):
    print(i)
    # SMALL DATASET
    # Using the generated frequencies, generate SS and DS LLR values of given size
    cv_noELUB_samples_LDS = np.random.choice(cv_noELUB_stretched_LLRs, size=n_ds_small, p=cv_noELUB_LDS_frequencies)
    cv_noELUB_samples_LSS = np.random.choice(cv_noELUB_stretched_LLRs, size=n_ss_small, p=cv_noELUB_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    cv_noELUB_DS = np.power(10, cv_noELUB_samples_LDS)
    cv_noELUB_SS = np.power(10, cv_noELUB_samples_LSS)
    cv_noELUB_small = calculate_metrics_all(cv_noELUB_SS, cv_noELUB_DS)

    # Using the generated frequencies, generate SS and DS LLR values of given size
    cv_ELUB_samples_LDS = np.random.choice(cv_ELUB_stretched_LLRs, size=n_ds_small, p=cv_ELUB_LDS_frequencies)
    cv_ELUB_samples_LSS = np.random.choice(cv_ELUB_stretched_LLRs, size=n_ss_small, p=cv_ELUB_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    cv_ELUB_DS = np.power(10, cv_ELUB_samples_LDS)
    cv_ELUB_SS = np.power(10, cv_ELUB_samples_LSS)
    cv_ELUB_small = calculate_metrics_all(cv_ELUB_SS, cv_ELUB_DS)

    # Using the generated frequencies, generate SS and DS LLR values of given size
    final_noELUB_samples_LDS = np.random.choice(final_noELUB_stretched_LLRs, size=n_ds_small, p=final_noELUB_LDS_frequencies)
    final_noELUB_samples_LSS = np.random.choice(final_noELUB_stretched_LLRs, size=n_ss_small,
                                                p=final_noELUB_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    final_noELUB_DS = np.power(10, final_noELUB_samples_LDS)
    final_noELUB_SS = np.power(10, final_noELUB_samples_LSS)
    final_noELUB_small = calculate_metrics_all(final_noELUB_SS, final_noELUB_DS)

    # Using the generated frequencies, generate SS and DS LLR values of given size
    final_ELUB_samples_LDS = np.random.choice(final_ELUB_stretched_LLRs, size=n_ds_small,
                                                p=final_ELUB_LDS_frequencies)
    final_ELUB_samples_LSS = np.random.choice(final_ELUB_stretched_LLRs, size=n_ss_small,
                                                p=final_ELUB_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    final_ELUB_DS = np.power(10, final_ELUB_samples_LDS)
    final_ELUB_SS = np.power(10, final_ELUB_samples_LSS)
    final_ELUB_small = calculate_metrics_all(final_ELUB_SS, final_ELUB_DS)

    # Using the generated frequencies, generate SS and DS LLR values of given size
    fig29_samples_LDS = np.random.choice(fig29_stretched_LLRs, size=n_ds_small, p=fig29_LDS_frequencies)
    fig29_samples_LSS = np.random.choice(fig29_stretched_LLRs, size=n_ss_small, p=fig29_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    fig29_DS = np.power(10, fig29_samples_LDS)
    fig29_SS = np.power(10, fig29_samples_LSS)
    fig29_small = calculate_metrics_all(fig29_SS, fig29_DS)

    # Using the generated frequencies, generate SS and DS LLR values of given size
    hulzen_samples_LDS = np.random.choice(hulzen_stretched_LLRs, size=n_ds_small, p=hulzen_LDS_frequencies)
    hulzen_samples_LSS = np.random.choice(hulzen_stretched_LLRs, size=n_ss_small, p=hulzen_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    hulzen_DS = np.power(10, hulzen_samples_LDS)
    hulzen_SS = np.power(10, hulzen_samples_LSS)
    hulzen_small = calculate_metrics_all(hulzen_SS, hulzen_DS)
    '''
    # Using the generated frequencies, generate SS and DS LLR values of given size
    benzine_samples_LDS = np.random.choice(benzine_stretched_LLRs, size=n_ds_small, p=benzine_LDS_frequencies)
    benzine_samples_LSS = np.random.choice(benzine_stretched_LLRs, size=n_ss_small, p=benzine_LSS_frequencies)

    # Transform to LR-values and calculate the metrics
    benzine_DS = np.power(10, benzine_samples_LDS)
    benzine_SS = np.power(10, benzine_samples_LSS)
    benzine_small = calculate_metrics_all(benzine_SS, benzine_DS)
'''
    # Generate normal data corresponding to a consistent LR-system
    normal_LDS = np.random.normal(-MU_ss, SIGMA, n_ds_small)
    normal_LSS = np.random.normal(MU_ss, SIGMA, n_ss_small)

    # Transform to LR-values and calculate the metrics
    normal_DS = np.power(math.e, normal_LDS)
    normal_SS = np.power(math.e, normal_LSS)
    normal_small = calculate_metrics_all(normal_SS, normal_DS)

    # Append metric values to corresponding arrays
    crossval_noELUB_small_dev.append(cv_noELUB_small[1])
    crossval_noELUB_small_cllr.append(cv_noELUB_small[0])

    crossval_ELUB_small_dev.append(cv_ELUB_small[1])
    crossval_ELUB_small_cllr.append(cv_ELUB_small[0])

    final_noELUB_small_dev.append(final_noELUB_small[1])
    final_noELUB_small_cllr.append(final_noELUB_small[0])

    final_ELUB_small_dev.append(final_ELUB_small[1])
    final_ELUB_small_cllr.append(final_ELUB_small[0])

    data_fig29_small_dev.append(fig29_small[1])
    data_fig29_small_cllr.append(fig29_small[0])

    hulzen_small_dev.append(hulzen_small[1])
    hulzen_small_cllr.append(hulzen_small[0])

    benzine_small_dev.append(benzine_small[1])
    benzine_small_cllr.append(benzine_small[0])

    normal_small_dev.append(normal_small[1])
    normal_small_cllr.append(normal_small[0])

    # MEDIUM DATASET
    # Repeat same process for medium dataset
    cv_noELUB_samples_LDS = np.random.choice(cv_noELUB_stretched_LLRs, size=n_ds_medium, p=cv_noELUB_LDS_frequencies)
    cv_noELUB_samples_LSS = np.random.choice(cv_noELUB_stretched_LLRs, size=n_ss_medium, p=cv_noELUB_LSS_frequencies)

    cv_noELUB_DS = np.power(10, cv_noELUB_samples_LDS)
    cv_noELUB_SS = np.power(10, cv_noELUB_samples_LSS)
    cv_noELUB_medium = calculate_metrics_all(cv_noELUB_SS, cv_noELUB_DS)

    cv_ELUB_samples_LDS = np.random.choice(cv_ELUB_stretched_LLRs, size=n_ds_medium, p=cv_ELUB_LDS_frequencies)
    cv_ELUB_samples_LSS = np.random.choice(cv_ELUB_stretched_LLRs, size=n_ss_medium, p=cv_ELUB_LSS_frequencies)

    cv_ELUB_DS = np.power(10, cv_ELUB_samples_LDS)
    cv_ELUB_SS = np.power(10, cv_ELUB_samples_LSS)
    cv_ELUB_medium = calculate_metrics_all(cv_ELUB_SS, cv_ELUB_DS)

    final_noELUB_samples_LDS = np.random.choice(final_noELUB_stretched_LLRs, size=n_ds_medium,
                                                p=final_noELUB_LDS_frequencies)
    final_noELUB_samples_LSS = np.random.choice(final_noELUB_stretched_LLRs, size=n_ss_medium,
                                                p=final_noELUB_LSS_frequencies)

    final_noELUB_DS = np.power(10, final_noELUB_samples_LDS)
    final_noELUB_SS = np.power(10, final_noELUB_samples_LSS)
    final_noELUB_medium = calculate_metrics_all(final_noELUB_SS, final_noELUB_DS)

    final_ELUB_samples_LDS = np.random.choice(final_ELUB_stretched_LLRs, size=n_ds_medium,
                                              p=final_ELUB_LDS_frequencies)
    final_ELUB_samples_LSS = np.random.choice(final_ELUB_stretched_LLRs, size=n_ss_medium,
                                              p=final_ELUB_LSS_frequencies)

    final_ELUB_DS = np.power(10, final_ELUB_samples_LDS)
    final_ELUB_SS = np.power(10, final_ELUB_samples_LSS)
    final_ELUB_medium = calculate_metrics_all(final_ELUB_SS, final_ELUB_DS)

    fig29_samples_LDS = np.random.choice(fig29_stretched_LLRs, size=n_ds_medium, p=fig29_LDS_frequencies)
    fig29_samples_LSS = np.random.choice(fig29_stretched_LLRs, size=n_ss_medium, p=fig29_LSS_frequencies)

    fig29_DS = np.power(10, fig29_samples_LDS)
    fig29_SS = np.power(10, fig29_samples_LSS)
    fig29_medium = calculate_metrics_all(fig29_SS, fig29_DS)

    hulzen_samples_LDS = np.random.choice(hulzen_stretched_LLRs, size=n_ds_medium, p=hulzen_LDS_frequencies)
    hulzen_samples_LSS = np.random.choice(hulzen_stretched_LLRs, size=n_ss_medium, p=hulzen_LSS_frequencies)

    hulzen_DS = np.power(10, hulzen_samples_LDS)
    hulzen_SS = np.power(10, hulzen_samples_LSS)
    hulzen_medium = calculate_metrics_all(hulzen_SS, hulzen_DS)

    benzine_samples_LDS = np.random.choice(benzine_stretched_LLRs, size=n_ds_medium, p=benzine_LDS_frequencies)
    benzine_samples_LSS = np.random.choice(benzine_stretched_LLRs, size=n_ss_medium, p=benzine_LSS_frequencies)

    benzine_DS = np.power(10, benzine_samples_LDS)
    benzine_SS = np.power(10, benzine_samples_LSS)
    benzine_medium = calculate_metrics_all(benzine_SS, benzine_DS)

    normal_LDS = np.random.normal(-MU_ss, SIGMA, n_ds_medium)
    normal_LSS = np.random.normal(MU_ss, SIGMA, n_ss_medium)

    normal_DS = np.power(math.e, normal_LDS)
    normal_SS = np.power(math.e, normal_LSS)
    normal_medium = calculate_metrics_all(normal_SS, normal_DS)

    crossval_noELUB_medium_dev.append(cv_noELUB_medium[1])
    crossval_noELUB_medium_cllr.append(cv_noELUB_medium[0])
    crossval_noELUB_medium_fid.append(cv_noELUB_medium[2])

    crossval_ELUB_medium_dev.append(cv_ELUB_medium[1])
    crossval_ELUB_medium_cllr.append(cv_ELUB_medium[0])
    crossval_ELUB_medium_fid.append(cv_ELUB_medium[2])

    final_noELUB_medium_dev.append(final_noELUB_medium[1])
    final_noELUB_medium_cllr.append(final_noELUB_medium[0])
    final_noELUB_medium_fid.append(final_noELUB_medium[2])

    final_ELUB_medium_dev.append(final_ELUB_medium[1])
    final_ELUB_medium_cllr.append(final_ELUB_medium[0])
    final_ELUB_medium_fid.append(final_ELUB_medium[2])

    data_fig29_medium_dev.append(fig29_medium[1])
    data_fig29_medium_cllr.append(fig29_medium[0])
    data_fig29_medium_fid.append(fig29_medium[2])

    hulzen_medium_dev.append(hulzen_medium[1])
    hulzen_medium_cllr.append(hulzen_medium[0])
    hulzen_medium_fid.append(hulzen_medium[2])

    benzine_medium_dev.append(benzine_medium[1])
    benzine_medium_cllr.append(benzine_medium[0])
    benzine_medium_fid.append(benzine_medium[2])

    normal_medium_dev.append(normal_medium[1])
    normal_medium_cllr.append(normal_medium[0])
    normal_medium_fid.append(normal_medium[2])

    # LARGE DATASET
    # Repeat same process for large dataset
    cv_noELUB_samples_LDS = np.random.choice(cv_noELUB_stretched_LLRs, size=n_ds_large, p=cv_noELUB_LDS_frequencies)
    cv_noELUB_samples_LSS = np.random.choice(cv_noELUB_stretched_LLRs, size=n_ss_large, p=cv_noELUB_LSS_frequencies)

    cv_noELUB_DS = np.power(10, cv_noELUB_samples_LDS)
    cv_noELUB_SS = np.power(10, cv_noELUB_samples_LSS)
    cv_noELUB_large = calculate_metrics_all(cv_noELUB_SS, cv_noELUB_DS)

    cv_ELUB_samples_LDS = np.random.choice(cv_ELUB_stretched_LLRs, size=n_ds_large, p=cv_ELUB_LDS_frequencies)
    cv_ELUB_samples_LSS = np.random.choice(cv_ELUB_stretched_LLRs, size=n_ss_large, p=cv_ELUB_LSS_frequencies)

    cv_ELUB_DS = np.power(10, cv_ELUB_samples_LDS)
    cv_ELUB_SS = np.power(10, cv_ELUB_samples_LSS)
    cv_ELUB_large = calculate_metrics_all(cv_ELUB_SS, cv_ELUB_DS)

    final_noELUB_samples_LDS = np.random.choice(final_noELUB_stretched_LLRs, size=n_ds_large,
                                                p=final_noELUB_LDS_frequencies)
    final_noELUB_samples_LSS = np.random.choice(final_noELUB_stretched_LLRs, size=n_ss_large,
                                                p=final_noELUB_LSS_frequencies)

    final_noELUB_DS = np.power(10, final_noELUB_samples_LDS)
    final_noELUB_SS = np.power(10, final_noELUB_samples_LSS)
    final_noELUB_large = calculate_metrics_all(final_noELUB_SS, final_noELUB_DS)

    final_ELUB_samples_LDS = np.random.choice(final_ELUB_stretched_LLRs, size=n_ds_large,
                                              p=final_ELUB_LDS_frequencies)
    final_ELUB_samples_LSS = np.random.choice(final_ELUB_stretched_LLRs, size=n_ss_large,
                                              p=final_ELUB_LSS_frequencies)

    final_ELUB_DS = np.power(10, final_ELUB_samples_LDS)
    final_ELUB_SS = np.power(10, final_ELUB_samples_LSS)
    final_ELUB_large = calculate_metrics_all(final_ELUB_SS, final_ELUB_DS)

    fig29_samples_LDS = np.random.choice(fig29_stretched_LLRs, size=n_ds_large, p=fig29_LDS_frequencies)
    fig29_samples_LSS = np.random.choice(fig29_stretched_LLRs, size=n_ss_large, p=fig29_LSS_frequencies)

    fig29_DS = np.power(10, fig29_samples_LDS)
    fig29_SS = np.power(10, fig29_samples_LSS)
    fig29_large = calculate_metrics_all(fig29_SS, fig29_DS)

    hulzen_samples_LDS = np.random.choice(hulzen_stretched_LLRs, size=n_ds_large, p=hulzen_LDS_frequencies)
    hulzen_samples_LSS = np.random.choice(hulzen_stretched_LLRs, size=n_ss_large, p=hulzen_LSS_frequencies)

    hulzen_DS = np.power(10, hulzen_samples_LDS)
    hulzen_SS = np.power(10, hulzen_samples_LSS)
    hulzen_large = calculate_metrics_all(hulzen_SS, hulzen_DS)

    benzine_samples_LDS = np.random.choice(benzine_stretched_LLRs, size=n_ds_large, p=benzine_LDS_frequencies)
    benzine_samples_LSS = np.random.choice(benzine_stretched_LLRs, size=n_ss_large, p=benzine_LSS_frequencies)

    benzine_DS = np.power(10, benzine_samples_LDS)
    benzine_SS = np.power(10, benzine_samples_LSS)
    benzine_large = calculate_metrics_all(benzine_SS, benzine_DS)

    normal_LDS = np.random.normal(-MU_ss, SIGMA, n_ds_large)
    normal_LSS = np.random.normal(MU_ss, SIGMA, n_ss_large)

    normal_DS = np.power(math.e, normal_LDS)
    normal_SS = np.power(math.e, normal_LSS)
    normal_large = calculate_metrics_all(normal_SS, normal_DS)

    crossval_noELUB_large_dev.append(cv_noELUB_large[1])
    crossval_noELUB_large_cllr.append(cv_noELUB_large[0])
    crossval_noELUB_large_fid.append(cv_noELUB_large[2])

    crossval_ELUB_large_dev.append(cv_ELUB_large[1])
    crossval_ELUB_large_cllr.append(cv_ELUB_large[0])
    crossval_ELUB_large_fid.append(cv_ELUB_large[2])

    final_noELUB_large_dev.append(final_noELUB_large[1])
    final_noELUB_large_cllr.append(final_noELUB_large[0])
    final_noELUB_large_fid.append(final_noELUB_large[2])

    final_ELUB_large_dev.append(final_ELUB_large[1])
    final_ELUB_large_cllr.append(final_ELUB_large[0])
    final_ELUB_large_fid.append(final_ELUB_large[2])

    data_fig29_large_dev.append(fig29_large[1])
    data_fig29_large_cllr.append(fig29_large[0])
    data_fig29_large_fid.append(fig29_large[2])

    hulzen_large_dev.append(hulzen_large[1])
    hulzen_large_cllr.append(hulzen_large[0])
    hulzen_large_fid.append(hulzen_large[2])

    benzine_large_dev.append(benzine_large[1])
    benzine_large_cllr.append(benzine_large[0])
    benzine_large_fid.append(benzine_large[2])

    normal_large_dev.append(normal_large[1])
    normal_large_cllr.append(normal_large[0])
    normal_large_fid.append(normal_large[2])

# Make one big array of all the devPAV values based on small datasets
smalls_dev = [crossval_ELUB_small_dev, data_fig29_small_dev, final_noELUB_small_dev, final_ELUB_small_dev, crossval_noELUB_small_dev, hulzen_small_dev, normal_small_dev]
smalls_dev = np.array(smalls_dev).flatten()

# Make one big array of all the Cllr values based on small datasets
smalls_cllr = [crossval_ELUB_small_cllr, data_fig29_small_cllr, final_noELUB_small_cllr, final_ELUB_small_cllr, crossval_noELUB_small_cllr, hulzen_small_cllr, normal_small_cllr]
smalls_cllr = np.array(smalls_cllr).flatten()

# Make one big array of all the devPAV values based on medium datasets
mediums_dev = [crossval_ELUB_medium_dev, data_fig29_medium_dev, final_noELUB_medium_dev, final_ELUB_medium_dev, crossval_noELUB_medium_dev, hulzen_medium_dev, normal_medium_dev]
mediums_dev = np.array(mediums_dev).flatten()

# Make one big array of all the Cllr values based on medium datasets
mediums_cllr = [crossval_ELUB_medium_cllr, data_fig29_medium_cllr, final_noELUB_medium_cllr, final_ELUB_medium_cllr, crossval_noELUB_medium_cllr, hulzen_medium_cllr, normal_medium_cllr]
mediums_cllr = np.array(mediums_cllr).flatten()

# Make one big array of all the Fid values based on medium datasets
mediums_fid = [crossval_ELUB_medium_fid, data_fig29_medium_fid, final_noELUB_medium_fid, final_ELUB_medium_fid, crossval_noELUB_medium_fid, hulzen_medium_fid, normal_medium_fid]
mediums_fid = np.array(mediums_fid).flatten()

# Make one big array of all the devPAV values based on large datasets
larges_dev = [crossval_ELUB_large_dev, data_fig29_large_dev, final_noELUB_large_dev, final_ELUB_large_dev, crossval_noELUB_large_dev, hulzen_large_dev, normal_large_dev]
larges_dev = np.array(larges_dev).flatten()

# Make one big array of all the Cllr values based on large datasets
larges_cllr = [crossval_ELUB_large_cllr, data_fig29_large_cllr, final_noELUB_large_cllr, final_ELUB_large_cllr, crossval_noELUB_large_cllr, hulzen_large_cllr, normal_large_cllr]
larges_cllr = np.array(larges_cllr).flatten()

# Make one big array of all the Fid values based on large datasets
larges_fid = [crossval_ELUB_large_fid, data_fig29_large_fid, final_noELUB_large_fid, final_ELUB_large_fid, crossval_noELUB_large_fid, hulzen_large_fid, normal_large_fid]
larges_fid = np.array(larges_fid).flatten()

# Combine results of dataset size comparison in dictionary
results_size_dev = {
    'Small': smalls_dev,
    'Medium': mediums_dev,
    'Large': larges_dev,
}
df_results = pd.DataFrame(results_size_dev)

# Determine overlap percentage
devs = np.array(list(results_size_dev.values()))
overlap_sizes_devs = average_overlap(devs)
print('Devpav sizes overlap:', overlap_sizes_devs)

# Plot distributions of dictionary
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)
for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('devPAV size dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Combine results of dataset size comparison in dictionary
results_size_cllr = {
    'Small': smalls_cllr,
    'Medium': mediums_cllr,
    'Large': larges_cllr,
}
df_results = pd.DataFrame(results_size_cllr)

# Determine overlap percentage
cllrs = np.array(list(results_size_cllr.values()))
overlap_sizes_cllrs = average_overlap(cllrs)
print('Cllr sizes overlap:', overlap_sizes_cllrs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)
for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('Cllr size dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Try because sometimes fid gives errors
try:
    # Combine results of dataset size comparison
    results_size_fid = {
        'Medium': mediums_fid,
        'Large': larges_fid,
    }
    df_results = pd.DataFrame(results_size_fid)

    # Compute overlap percentage
    fids = np.array(list(results_size_fid.values()))
    overlap_sizes_fids = average_overlap(fids)
    print('Fids overlap:', overlap_sizes_fids)

    # Plot results
    fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)
    for i, column in enumerate(df_results.columns):
        sns.violinplot(data=df_results[column], ax=axes[i])
        axes[i].set_title(column)
    plt.suptitle('Fid size dataset comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")

# Collect results of devPAV on all small datasets
results_dataset_dev_small = {
    '1': crossval_ELUB_small_dev,
    '2': crossval_noELUB_small_dev,
    '3': final_ELUB_small_dev,
    '4': final_noELUB_small_dev,
    '5': data_fig29_small_dev,
    '6': hulzen_small_dev,
    #'7': benzine_small_dev,
    'Normal data': normal_small_dev,
}
df_results = pd.DataFrame(results_dataset_dev_small)

# Compute overlap percentage
devs = np.array(list(results_dataset_dev_small.values()))
overlap_datasets_devs = average_overlap(devs)
print('Devpav overlap small:', overlap_datasets_devs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('devPAV small dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of devPAV on all medium datasets
results_dataset_dev_medium = {
    '1': crossval_ELUB_medium_dev,
    '2': crossval_noELUB_medium_dev,
    '3': final_ELUB_medium_dev,
    '4': final_noELUB_medium_dev,
    '5': data_fig29_medium_dev,
    '6': hulzen_medium_dev,
    #'7': benzine_medium_dev,
    'Normal data': normal_medium_dev,
}
df_results = pd.DataFrame(results_dataset_dev_medium)

# Compute overlap percentage
devs = np.array(list(results_dataset_dev_medium.values()))
overlap_datasets_devs = average_overlap(devs)
print('Devpav overlap medium:', overlap_datasets_devs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('devPAV medium dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of devPAV on all large datasets
results_dataset_dev_large = {
    '1': crossval_ELUB_large_dev,
    '2': crossval_noELUB_large_dev,
    '3': final_ELUB_large_dev,
    '4': final_noELUB_large_dev,
    '5': data_fig29_large_dev,
    '6': hulzen_large_dev,
    #'7': benzine_large_dev,
    'Normal data': normal_large_dev,
}
df_results = pd.DataFrame(results_dataset_dev_large)

# Compute overlap percentage
devs = np.array(list(results_dataset_dev_large.values()))
overlap_datasets_devs = average_overlap(devs)
print('Devpav overlap large:', overlap_datasets_devs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('devPAV large dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of cllr on all small datasets
results_dataset_cllr_small = {
    '1': crossval_ELUB_small_cllr,
    '2': crossval_noELUB_small_cllr,
    '3': final_ELUB_small_cllr,
    '4': final_noELUB_small_cllr,
    '5': data_fig29_small_cllr,
    '6': hulzen_small_cllr,
   # '7': benzine_small_cllr,
    'Normal data': normal_small_cllr,
}
df_results = pd.DataFrame(results_dataset_cllr_small)

# Compute overlap percentage
cllrs = np.array(list(results_dataset_cllr_small.values()))
overlap_datasets_cllrs = average_overlap(cllrs)
print('Cllr overlap small:', overlap_datasets_cllrs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('Cllr small dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of Cllr on all medium datasets
results_dataset_cllr_medium = {
    '1': crossval_ELUB_medium_cllr,
    '2': crossval_noELUB_medium_cllr,
    '3': final_ELUB_medium_cllr,
    '4': final_noELUB_medium_cllr,
    '5': data_fig29_medium_cllr,
    '6': hulzen_medium_cllr,
    #'7': benzine_medium_cllr,
    'Normal data': normal_medium_cllr,
}
df_results = pd.DataFrame(results_dataset_cllr_medium)

# Compute overlap percentage
cllrs = np.array(list(results_dataset_cllr_medium.values()))
overlap_datasets_cllrs = average_overlap(cllrs)
print('Cllr overlap medium:', overlap_datasets_cllrs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('Cllr medium dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Collect results of cllr on all large datasets
results_dataset_cllr_large = {
    '1': crossval_ELUB_large_cllr,
    '2': crossval_noELUB_large_cllr,
    '3': final_ELUB_large_cllr,
    '4': final_noELUB_large_cllr,
    '5': data_fig29_large_cllr,
    '6': hulzen_large_cllr,
   # '7': benzine_large_cllr,
    'Normal data': normal_large_cllr,
}
df_results = pd.DataFrame(results_dataset_cllr_large)

# Compute overlap percentage
cllrs = np.array(list(results_dataset_cllr_large.values()))
overlap_datasets_cllrs = average_overlap(cllrs)
print('Cllr overlap large:', overlap_datasets_cllrs)

# Plot results
fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

for i, column in enumerate(df_results.columns):
    sns.violinplot(data=df_results[column], ax=axes[i])
    axes[i].set_title(column)
plt.suptitle('Cllr large dataset comparison', fontsize=16)
plt.tight_layout()
plt.show()

try:
    # Collect results of Fid on all medium datasets
    results_dataset_fid_medium = {
        '1': crossval_ELUB_medium_fid,
        '2': crossval_noELUB_medium_fid,
        '3': final_ELUB_medium_fid,
        '4': final_noELUB_medium_fid,
        '5': data_fig29_medium_fid,
        '6': hulzen_medium_fid,
       # '7': benzine_medium_fid,
        'Normal data': normal_medium_fid,
    }
    df_results = pd.DataFrame(results_dataset_fid_medium)

    # Compute overlap percentage
    fids = np.array(list(results_dataset_fid_medium.values()))
    overlap_datasets_fids = average_overlap(fids)
    print('Fid overlap medium:', overlap_datasets_fids)

    # Plot reuslts
    fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

    for i, column in enumerate(df_results.columns):
        sns.violinplot(data=df_results[column], ax=axes[i])
        axes[i].set_title(column)
    plt.suptitle('Fid medium dataset comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")

try:
    # Collect results of devPAV on all large datasets
    results_dataset_fid_large = {
        '1': crossval_ELUB_large_fid,
        '2': crossval_noELUB_large_fid,
        '3': final_ELUB_large_fid,
        '4': final_noELUB_large_fid,
        '5': data_fig29_large_fid,
        '6': hulzen_large_fid,
       # '7': benzine_large_fid,
        'Normal data': normal_large_fid,
    }
    df_results = pd.DataFrame(results_dataset_fid_large)

    # Compute overlap percentage
    fids = np.array(list(results_dataset_fid_large.values()))
    overlap_datasets_fids = average_overlap(fids)
    print('Fid overlap large:', overlap_datasets_fids)

    # Plot results
    fig, axes = plt.subplots(ncols=len(df_results.columns), figsize=(15, 6), sharey=True)

    for i, column in enumerate(df_results.columns):
        sns.violinplot(data=df_results[column], ax=axes[i])
        axes[i].set_title(column)
    plt.suptitle('Fid large dataset comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")