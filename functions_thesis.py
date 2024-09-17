import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import gaussian_kde, norm
import lir
from scipy.interpolate import interp1d
from scipy.integrate import simps

# FUNCTIONS FOR DEVPAV

# this function is copied from the lir library
def _calcsurface(c1: (float, float), c2: (float, float)) -> float:
    """
    Helper function that calculates the desired surface for two xy-coordinates
    """
    # step 1: calculate intersection (xs, ys) of straight line through coordinates with identity line (if slope (a) = 1, there is no intersection and surface of this parrellogram is equal to deltaY * deltaX)
    x1, y1 = c1
    x2, y2 = c2
    a = (y2 - y1) / (x2 - x1)

    if a == 1:
        # dan xs equals +/- Infinite en is er there is no intersection with the identity line

        # the surface of the parallellogram is:
        surface = (x2 - x1) * np.abs(y1 - x1)

    elif (a < 0):
        raise ValueError(f"slope is negative; impossible for PAV-transform. Coordinates are {c1} and {c2}. Calculated slope is {a}")
    else:
        # than xs is finite:
        b = y1 - a * x1
        xs = b / (1 - a)
        # xs

        # step 2: check if intersection is located within line segment c1 and c2.
        if x1 < xs and x2 >= xs:
            # then intersection is within
            # (situation 1 of 2) if y1 <= x1 than surface is:
            if y1 <= x1:
                surface = 0.5 * (xs - y1) * (xs - x1) - 0.5 * (xs - x1) * (xs - x1) + 0.5 * (y2 - xs) * (x2 - xs) - 0.5 * (
                            x2 - xs) * (x2 - xs)
            else:
                # (situation 2 of 2) than y1 > x1, and surface is:
                surface = 0.5 * (xs - x1) ** 2 - 0.5 * (xs - y1) * (xs - x1) + 0.5 * (x2 - xs) ** 2 - 0.5 * (x2 - xs) * (
                            y2 - xs)
                # dit is the same as 0.5 * (xs - x1) * (xs - y1) - 0.5 * (xs - y1) * (xs - y1) + 0.5 * (y2 - xs) * (x2 - xs) - 0.5 * (y2 - xs) * (y2 - xs) + 0.5 * (y1 - x1) * (y1 - x1) + 0.5 * (x2 - y2) * (x2 -y2)
        else:  # then intersection is not within line segment
            # if (situation 1 of 4) y1 <= x1 AND y2 <= x1, and surface is
            if y1 <= x1 and y2 <= x1:
                surface = 0.5 * (y2 - y1) * (x2 - x1) + (x1 - y2) * (x2 - x1) + 0.5 * (x2 - x1) * (x2 - x1)
            elif y1 > x1:  # (situation 2 of 4) then y1 > x1, and surface is
                surface = 0.5 * (x2 - x1) * (x2 - x1) + (y1 - x2) * (x2 - x1) + 0.5 * (y2 - y1) * (x2 - x1)
            elif y1 <= x1 and y2 > x1:  # (situation 3 of 4). This should be the last possibility.
                surface = 0.5 * (y2 - y1) * (x2 - x1) - 0.5 * (y2 - x1) * (y2 - x1) + 0.5 * (x2 - y2) * (x2 - y2)
            else:
                # situation 4 of 4 : this situation should never appear. There is a fourth sibution as situation 3, but than above the identity line. However, this is impossible by definition of a PAV-transform (y2 > x1).
                raise ValueError(f"unexpected coordinate combination: ({x1}, {y1}) and ({x2}, {y2})")
    return surface

# this function is taken from the lir library
def _devpavcalculator(lrs, pav_lrs, y):
    """
    Function that calculates davPAV for a PAVresult for SSLRs and DSLRs.

    Input:
    - Lrs: np.array with LR-values.
    - pav_lrs: np.array with LRs after PAV-transform. y = np.array with labels (1 for H1 and 0 for H2).

    Output:
    - devPAV value.

    """
    DSLRs, SSLRs = lir.Xy_to_Xn(lrs, y)
    DSPAVLRs, SSPAVLRs = lir.Xy_to_Xn(pav_lrs, y)
    PAVresult = np.concatenate([SSPAVLRs, DSPAVLRs])
    Xen = np.concatenate([SSLRs, DSLRs])

    # order coordinates based on x's then y's and filtering out identical datapoints
    data = np.unique(np.array([Xen, PAVresult]), axis=1)
    Xen = data[0, :]
    Yen = data[1, :]

    # pathological cases
    # first one of four: PAV-transform has a horizonal line to log(X) = -Inf as to log(X) = Inf
    if Yen[0] != 0 and Yen[-1] != np.inf and Xen[-1] == np.inf and Xen[-1] == np.inf:
        return np.Inf

    # second of four: PAV-transform has a horizontal line to log(X) = -Inf
    if Yen[0] != 0 and Xen[0] == 0 and Yen[-1] == np.inf:
        return np.Inf

    # third of four: PAV-transform has a horizontal line to log(X) = Inf
    if Yen[0] == 0 and Yen[-1] != np.inf and Xen[-1] == np.inf:
        return np.Inf

    # fourth of four: PAV-transform has one vertical line from log(Y) = -Inf to log(Y) = Inf
    wh = (Yen == 0) | (Yen == np.inf)
    if np.sum(wh) == len(Yen):
        return np.nan

    else:
        # then it is not a  pathological case with weird X-values and devPAV can be calculated

        # filtering out -Inf or 0 Y's
        wh = (Yen > 0) & (Yen < np.inf)
        Xen = np.log10(Xen[wh])
        Yen = np.log10(Yen[wh])
        # create an empty list with size (len(Xen))
        devPAVs = [None] * len(Xen)
        # sanity check
        if len(Xen) == 0:
            return np.nan
        elif len(Xen) == 1:
            return abs(Xen - Yen)
        # than calculate devPAV
        else:
            deltaX = Xen[-1] - Xen[0]
            surface = 0
            for i in range(1, (len(Xen))):
                surface = surface + _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
                devPAVs[i - 1] = _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
            # return(list(surface/a, PAVresult, Xen, Yen, devPAVs))
            return surface / deltaX


def scaled_devpavcalc(lrs, pav_lrs, y):
    """
    Function that calculates the scaled davPAV for a PAVresult for SSLRs and DSLRs.

    Input:
    - lrs: np.array with LR-values.
    - pav_lrs: np.array with LRs after PAV-transform.
    - y: np.array with labels (1 for H1 and 0 for H2).

    Output:
    - scaled devPAV value.

    """
    DSLRs, SSLRs = lir.Xy_to_Xn(lrs, y)
    DSPAVLRs, SSPAVLRs = lir.Xy_to_Xn(pav_lrs, y)
    PAVresult = np.concatenate([SSPAVLRs, DSPAVLRs])
    Xen = np.concatenate([SSLRs, DSLRs])

    # Order coordinates based on x's then y's and filtering out identical datapoints
    data = np.unique(np.array([Xen, PAVresult]), axis=1)
    Xen = data[0, :]
    Yen = data[1, :]

    # pathological cases
    # first one of four: PAV-transform has a horizonal line to log(X) = -Inf as to log(X) = Inf
    if Yen[0] != 0 and Yen[-1] != np.inf and Xen[-1] == np.inf and Xen[-1] == np.inf:
        return np.Inf

    # second of four: PAV-transform has a horizontal line to log(X) = -Inf
    if Yen[0] != 0 and Xen[0] == 0 and Yen[-1] == np.inf:
        return np.Inf

    # third of four: PAV-transform has a horizontal line to log(X) = Inf
    if Yen[0] == 0 and Yen[-1] != np.inf and Xen[-1] == np.inf:
        return np.Inf

    # forth of four: PAV-transform has one vertical line from log(Y) = -Inf to log(Y) = Inf
    wh = (Yen == 0) | (Yen == np.inf)
    if np.sum(wh) == len(Yen):
        return np.nan

    else:
        # then it is not a  pathological case with weird X-values and devPAV can be calculated

        # filtering out -Inf or 0 Y's
        wh = (Yen > 0) & (Yen < np.inf)
        Xen = np.log10(Xen[wh])
        Yen = np.log10(Yen[wh])
        # create an empty list with size (len(Xen))
        devPAVs = [None] * len(Xen)
        # sanity check
        if len(Xen) == 0:
            return np.nan
        elif len(Xen) == 1:
            return abs(Xen - Yen)
        # then calculate devPAV
        else:
            # determine the difference in x-values and y-values
            deltaX = Xen[-1] - Xen[0]
            deltaY = Yen[-1] - Yen[0]
            surface = 0
            for i in range(1, (len(Xen))):
                surface = surface + _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
                devPAVs[i - 1] = _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
            # scale by surface
            return surface / (deltaX*deltaY)

def devpav(lrs: np.ndarray, y: np.ndarray) -> float:
    """
    Function that calculates normal devPAV for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - devPAV value.
    """
    # Check if input is valid
    if sum(y) == len(y) or sum(y) == 0:
        raise ValueError('devpav: illegal input: at least one value is required for each class')

    # Determine pav lrs
    cal = lir.IsotonicCalibrator()
    pavlrs = cal.fit_transform(lrs, y)

    # Return devpav
    return _devpavcalculator(lrs, pavlrs, y)

def scaled_devpav(lrs: np.ndarray, y: np.ndarray) -> float:
    """
    Function that calculates scaled devPAV for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - scaled devPAV value.
    """

    # Check if input is valid
    if sum(y) == len(y) or sum(y) == 0:
        raise ValueError('devpav: illegal input: at least one value is required for each class')

    # Determine pav lrs
    cal = lir.IsotonicCalibrator()
    pavlrs = cal.fit_transform(lrs, y)

    # Return scaled devpav
    return scaled_devpavcalc(lrs, pavlrs, y)

def devpav_new(lrs: np.ndarray, y:np.ndarray) -> float:
    """
    Function that calculates smoothed devPAV for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - smoothed devPAV value.
    """
    # determine pav lrs
    cal = lir.IsotonicCalibrator()
    pavlrs = cal.fit_transform(lrs, y)

    # get original and pav lrs and ensure row vector
    x = np.ravel(lrs)
    y = np.ravel(pavlrs)

    # calculating devPAV only makes sense if the original and transformed
    # variables have the same domain; in this case they are both LRs with a
    # domain between 0 and +inf.
    if any(x < 0) or any(y < 0):
        raise ValueError('Both variables should be non-negative.')

    # Convert both coordinates to log10
    x = np.log10(x)
    y = np.log10(y)

    # Sort values
    x = np.sort(x)
    y = np.sort(y)

    # Exclude datapoints with one or two non-finite coordinates
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    # Add initial and final points at the identity line
    x = np.concatenate(([x[0]], x, [x[-1]]))
    y = np.concatenate(([x[0]], y, [x[-1]]))

    # Rotate the transformation line clockwise by 45 degrees
    x_rot = (x + y) / np.sqrt(2)
    y_rot = (y - x) / np.sqrt(2)

    # Add new points to the line, where it crosses the (new rotated) X-axis.
    # This is when the Y-values of two adjacent points have opposite signs.
    i_cross = np.where(np.abs(np.diff(np.sign(y_rot))) == 2)[0]
    # Add new points in backwards order, so the cross indices are unchanged
    for i_p in range(len(i_cross) - 1, -1, -1):
        i_c = i_cross[i_p]
        x_dif = np.diff(x_rot[i_c:i_c + 2])
        y_dif = np.diff(y_rot[i_c:i_c + 2])

        # The added x-coordinate is shifted proportional to the y-values
        x_add = x_rot[i_c] + x_dif * np.abs(y_rot[i_c] / y_dif)
        x_add = np.array([x_add]).reshape(1, )
        x_rot = np.concatenate((x_rot[:i_c + 1], x_add, x_rot[i_c + 1:]))
        y_add = 0
        y_rot = np.concatenate((y_rot[:i_c + 1], [y_add], y_rot[i_c + 1:]))

    # Determine corners of step function
    critical_points = []
    critical_points.append((x_rot[0], y_rot[0]))
    increasing = True
    for i in range(1, len(y_rot)):
        if increasing and y_rot[i] < y_rot[i - 1]:
            # Transition from increasing to decreasing
            critical_points.append((x_rot[i-1], y_rot[i-1]))
            increasing = False
        elif not increasing and y_rot[i] > y_rot[i - 1]:
            # Transition from decreasing to increasing
            critical_points.append((x_rot[i - 1], y_rot[i - 1]))
            increasing = True
    critical_points.append((x_rot[-1], y_rot[-1]))

    # Determine the triangles that form the steps (corners and middle point)
    tuples = []
    for i in range(0, len(critical_points) - 2, 2):
        tuples.append([critical_points[i], critical_points[i+1], critical_points[i+2]])

    # Determine all the lines of the step function
    lines = []
    for i in range(0, len(critical_points)-1, 1):
        line = [(critical_points[i][0],critical_points[i][1]), (critical_points[i+1][0], critical_points[i+1][1])]
        lines.append(line)

    # Determine the new points we want to interpolate linearly: corners stay, middle points are cut off by
    # determining middle points of lines from corner to middle point of triangle and drawing line between
    # middle points
    points = []
    points.append(lines[0][0])
    for line in lines:
        mid_x = (line[0][0] + line[1][0])/2
        mid_y = (line[0][1] + line[1][1])/2
        points.append((mid_x, mid_y))
    points.append((lines[-1][1]))

    # Determine x-values and y-values of points
    xvals = np.array([point[0] for point in points])
    yvals = np.array([point[1] for point in points])

    # Interpolate linearly between the points, giving a piecewise linear function
    interp_func = interp1d(xvals, yvals, kind='linear')

    # Obtain corresponding y-values from the interpolation function
    new_x = np.linspace(min(xvals), max(xvals), 100)
    new_y = interp_func(new_x)

    # Determine area
    area = np.diff(new_x) * np.abs(new_y[:-1] + new_y[1:]) / 2
    total_area = np.sum(area)

    # Determine smoothed devPAV
    x_range = np.max(new_x) - np.min(new_x)
    smoothed_devpav = total_area / x_range

    return smoothed_devpav

# FUNCTIONS FOR CLLR

def cllr(lrs, y):
    """
    Function that calculates cllr cal using logarithmic scoring rule for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - cllr cal value.
    """

    # Determine total cllr and discrimination power and subtract to find cllr cal
    cllrmax = lir.metrics.cllr(lrs, y)
    cllrmin = lir.metrics.cllr_min(lrs, y)

    return cllrmax - cllrmin


def brier(lrs, y):
    """
    Function that calculates cllr cal using brier score for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - cllr cal value with brier score.
    """

    # Make dictionary of labels and corresponding LR-values
    grouped_LR = {}
    for label, LR in zip(y, lrs):
        if label not in grouped_LR:
            grouped_LR[label] = []
        grouped_LR[label].append(LR)

    l1 = len(grouped_LR.get(1, []))
    l2 = len(grouped_LR.get(0, []))
    sum_1 = 0
    sum_2 = 0
    for label, LR_list in grouped_LR.items():
        # Determine Brier scores for labels using posterior
        # H_p true
        if label == 1:
            for LR in LR_list:
                if LR != 0 and LR != np.inf:
                    posterior = LR / (1 + LR)
                    sum_1 += ((posterior - 1) ** 2)
        # H_d true
        else:
            for LR in LR_list:
                if LR != 0 and LR != np.inf:
                    posterior = LR / (1 + LR)
                    sum_2 += (posterior ** 2)
    # Determine ECE using Brier score
    brier = 0.5 / l1 * sum_1 + 0.5 / l2 * sum_2

    return brier

def zero_one(LRs, labels):
    """
    Function that calculates cllr using zero-one score for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - cllr cal value with zero-one score.
    """

    n = len(LRs)

    # Count misclassifications by determining posteriors
    misclas = 0
    misclas2 = 0

    for i in range(n):
        if LRs[i] != 0 and LRs[i] != np.inf:
            posterior = LRs[i] / (1 + LRs[i])
            if posterior > 0.5 and labels[i] == 0:
                misclas += 1
            elif posterior < 0.5 and labels[i] == 1:
                misclas += 1

    # Determine frequency of misclassifications
    misclas = misclas / n

    return misclas

def spherical(LRs, labels):
    """
    Function that calculates cllr using spherical scoring rule for LR data under H1 and H2.

    Input:
    - lrs: np.array with LR-values.
    - y: np.array with labels, 1 for H1 and 0 for H2.

    Output:
    - cllr cal value with spherical scoring rule.
    """
    n = len(LRs)
    spherical = 0
    m = 0

    # Determine ECE using spherical scoring rule
    for i in range(n):
        if LRs[i] != 0 and LRs[i] != np.inf:
            posterior = LRs[i] / (1 + LRs[i])
            spherical += (labels[i] * posterior + (1 - labels[i]) * (1 - posterior)) / sqrt(
                posterior ** 2 + (1 - posterior) ** 2)
            m += 1

    # To avoid error
    if m == 0:
        m = 1

    return spherical / m

# FUNCTIONS FOR FIDUCIAL METRICS

# This function is based on Jan Hannig's R-code
def fiducial_sample(data,nfid):
    """
    Function that makes fiducial samples of data.

    Input:
    - data: np.array of LR-values.
    - nfid = amount of fiducial samples.

    Output:
    - Dictionary that contains the following keys:
        - 'data': sorted data,
        - 'u': fiducial samples,
        - 'n': number of data points,
        - 'nfid': number of fiducial samples.
    """
    n = len(data)

    # Sort data
    sorted_data = np.sort(data)
    sorted_data = np.transpose(sorted_data)

    # Make nfid fiducial samples  of length ndata
    u = np.sort([np.random.uniform(size=n) for _ in range(nfid)])
    u = np.transpose(u)
    u = u[::-1]

    return {
        'data': sorted_data,  # Sorted data
        'u': u,  # Fiducial samples
        'n': n,  # Number of data points
        'nfid': nfid  # Number of fiducial samples
    }

# This function is based on Jan Hannigs R-code
def particle_grid(xgrid, lrt_fsample):
    """
    Function that defines grid for fiducial inference and calculates survival functions and integral on grid.

    Input:
    - xgrid = grid for x-values,
    - lrt_fsample = dictionary with following keys:
        - 'data': sorted data,
        - 'u': fiducial samples,
        - 'n': number of data points,
        - 'nfid': number of fiducial samples.
    Output:
    - Dictionary with following keys:
        - 'grid': grid,
        - 'survival': array of survival function values at each grid point for each fiducial sample,
        - 'bottom': array representing the integral of the survival function up to each grid point for each fiducial sample
        - 'nfid': number of fiducial samples,
        - 'n': number of data points.
    """

    # Sort grid points
    ngrid = len(xgrid)
    grid = np.sort(xgrid)

    # Extract and prepare the data and fiducial sample information
    data = np.concatenate(([0],lrt_fsample['data'], [np.inf]))
    n = lrt_fsample['n']
    nfid = lrt_fsample['nfid']

    # Initialize arrays for survival functions and integrals
    both_integrals_survival = []
    both_integrals_bottom = []

    for i in range(nfid):
        # Each fiducial sample processed separately
        u = np.concatenate(([1], lrt_fsample['u'][:, i], [0]))
        u = u.reshape(-1,1)
        data = data.reshape(-1,1)

        # Calculate the expression results based on fiducial sample and data
        expression_result = u[n] * (data[n] - data[n-1] +(1 / n))
        expression_result = expression_result.reshape(-1,1)

        # Concatenate and compute the integral of survival function
        concatenated = np.concatenate((expression_result, (data[n:0:-1]*u[n-1::-1] - data[n-1::-1]*u[n:0:-1])/2))
        # Cumulative sum to get the integral
        dataint = np.cumsum(concatenated)
        # Flip integral values for correct alignment
        flipped_int = np.flip(dataint)
        flipped_int = flipped_int.reshape(-1,1)
        dataintegral = flipped_int + (data[n]*u[n]-data[0:(n+1)]*u[0:(n+1)])/2
        dataintegral_f = dataintegral.flatten().tolist()

        # Initialize arrays for survival function and integral results
        survival_array = np.empty(ngrid)
        integral_array = np.empty(ngrid)

        # Evaluate survival and integral values at each grid point
        for j in range(ngrid):
        # Find the indices of data points that are nearest to the grid point
            indeces_ub = np.where(data>=grid[j])
            index_ub = indeces_ub[0][0]
            indeces_lb = np.where(data <= grid[j])
            index_lb = indeces_lb[0][-1]
            # If exactly hitting a grid point
            if index_ub <= index_lb:
                survival_array[j] = u[index_lb]
                integral_array[j] = dataintegral_f[index_lb]
            # If grid point is beyond the range of the data
            elif index_ub == n+1:
                gridoff = np.exp(-(grid[j] - data[n]) / (data[n] - data[n-1] + (1 / n)))
                survival_array[j] = u[n] * gridoff
                integral_array[j] = np.nan
            # If grid point is between data points
            else:
                survival_array[j] = (u[index_ub] * (grid[j] - data[index_lb]) + u[index_lb] *
                                        (data[index_ub] - grid[j]))/(data[index_ub] - data[index_lb])
                integral_array[j] = dataintegral_f[index_ub] + \
                                       (data[index_ub] - grid[j]) * (survival_array[j] + u[index_ub]) / 2

        # Append the results for the current fiducial sample
        both_integrals_survival.append(survival_array)
        both_integrals_bottom.append(grid*survival_array + integral_array)

    return {
        'grid': grid,
        'survival': np.transpose(np.array(both_integrals_survival)),
        'bottom': np.transpose(np.array(both_integrals_bottom)),
        'nfid': lrt_fsample['nfid'],
        'n': lrt_fsample['n']
    }

# This function is based on Jan Hannig's R-code
def fid_diff_log(particle_top, particle_bottom, coarse_index=None):
    """
    Function to calculate the log difference of fiducial sample values between two particles.

    Input:
    - particle_top: dictionary containing fiducial sample data for the top particle with keys:
       - 'grid': grid of x-values,
       - 'survival': survival function values for each fiducial sample,
       - 'nfid': number of fiducial samples,
       - 'n': number of data points.
    - particle_bottom: dictionary containing fiducial sample data for the bottom particle with the same keys as particle_top.
    - coarse_index: optional array of indices to coarsely sample the grid. If None, uses all indices.

    Output:
    - Dictionary with the following keys:
       - 'fsample_top': survival function values for the top particle,
       - 'fsample_bottom': bottom function values for the bottom particle,
       - 'n_top': number of data points for the top particle,
       - 'n_bottom': number of data points for the bottom particle,
       - 'nfid': number of fiducial samples,
       - 'fdiff_logratio': logarithm of the ratio of differences between the top and bottom fiducial samples,
       - 'grid': grid of x-values,
       - 'dgrid': coarse grid used for sampling.
       """

    grid = particle_top['grid']

    # Check if grids and number of fiducial samples match
    if np.sum(grid != particle_bottom['grid']) > 0 or particle_bottom['nfid'] != particle_top['nfid']:
        print('Mismatch of inputs')
        return None

    # Use all indices if coarse_index is not provided or is invalid
    if coarse_index is None or len(coarse_index) <= 1:
        coarse_index = np.arange(0, len(grid) + 1)

    # Find intersection of coarse_index with valid grid indices
    cindex = np.intersect1d(np.arange(0, len(grid) + 1), coarse_index)

    if len(cindex) < 2:
        print('Incompatible coarse_index')
        return None

    # Extract fiducial samples
    fidTop = particle_top['survival']
    fidBottom = particle_bottom['bottom']

    # Calculate the difference in fiducial samples
    d_fid_Top = -np.diff(fidTop[cindex], axis=0)
    d_fid_Bottom = -np.diff(fidBottom[cindex], axis=0)

    # Compute the log difference between the top and bottom fiducial samples
    fid_sample_slope_ratio = (np.log10(d_fid_Top) - np.log10(d_fid_Bottom))
    coarsegrid = grid[cindex]

    return {
        'fsample_top': fidTop,
        'fsample_bottom': fidBottom,
        'n_top': particle_top['n'],
        'n_bottom': particle_bottom['n'],
        'nfid': particle_top['nfid'],
        'fdiff_logratio': fid_sample_slope_ratio,
        'grid': grid,
        'dgrid': coarsegrid
    }

# This function is based on Jan Hannig's code
def fid_diff_CI(fid_dif_sample, alpha=0.05):
    """
    Function to calculate confidence intervals for the fiducial differences.

    Input:
    - fid_dif_sample: dictionary containing fiducial differences with keys:
        - 'fdiff_logratio': logarithmic differences of fiducial sample ratios,
        - 'dgrid': coarse grid used for sampling.
    - alpha: significance level for confidence interval (default is 0.05 for 95% CI).

    Output:
    - Dictionary with the following keys:
        - 'mean': central value of the fiducial slope,
        - 'uniform_lower': lower bound of the uniform confidence interval,
        - 'uniform_upper': upper bound of the uniform confidence interval,
        - 'median': median of the fiducial slope,
        - 'point_lower': lower bound of the pointwise confidence interval,
        - 'point_upper': upper bound of the pointwise confidence interval,
        - 'dgrid': coarse grid used for sampling.
        """

    fiducial_slope = fid_dif_sample['fdiff_logratio']

    # Calculate the central quantile of the fiducial slope
    CI_center = np.apply_along_axis(lambda x: np.quantile(x, 0.5, axis=0, keepdims=True), 1, fiducial_slope)

    # Calculate the scale of the confidence interval
    CI_scale = np.mean(np.abs(fiducial_slope - CI_center), axis=1, keepdims=True)

    # Calculate the scaled fiducial differences
    fid_diff = fiducial_slope - CI_center
    fid_scaled_diff = fid_diff / CI_scale
    fid_abs_scaled_diff = np.abs(fid_scaled_diff)

    # Compute the maximum of the scaled fiducial differences
    fid_max = np.nanmax(fid_abs_scaled_diff, axis=0)

    # Calculate the cutoff value for the confidence intervals
    cut_off = np.quantile(fid_max, 1 - alpha, axis=0)

    # # Calculate the confidence intervals
    mean = CI_center
    uniform_lower = CI_center - cut_off * CI_scale
    uniform_upper = CI_center + cut_off * CI_scale
    median = CI_center
    point_lower = np.apply_along_axis(lambda x: np.quantile(x, alpha / 2, axis=0, keepdims=True), 1, fiducial_slope)
    point_upper = np.apply_along_axis(lambda x: np.quantile(x, 1 - alpha / 2, axis=0, keepdims=True), 1, fiducial_slope)

    return {
        'mean': mean,
        'uniform_lower': uniform_lower,
        'uniform_upper': uniform_upper,
        'median': median,
        'point_lower': point_lower,
        'point_upper': point_upper,
        'dgrid': fid_dif_sample['dgrid'],
    }

def compare_CI(compare_sample, alpha=0.05):

    """
    Function to calculate confidence intervals for comparison samples.

    Input:
    - compare_sample: a 2D numpy array where each row represents a sample and each column represents a different comparison.
    - alpha: significance level for the confidence interval (default is 0.05 for 95% CI).

    Output:
    - Dictionary with the following keys:
        - 'mean': central value of the confidence interval (median of the samples),
        - 'uniform_lower': lower bound of the uniform confidence interval,
        - 'uniform_upper': upper bound of the uniform confidence interval,
        - 'median': median of the comparison samples,
        - 'point_lower': lower bound of the pointwise confidence interval,
        - 'point_upper': upper bound of the pointwise confidence interval.
    """

    # Calculate the center of the confidence interval
    CI_center = np.apply_along_axis(np.quantile, 1, compare_sample, 0.5, na_rm=True)

    # Calculate the scale of the confidence interval
    CI_scale = np.mean(np.abs(compare_sample - CI_center), axis=1)

    # Calculate fid_max (maximum of the scaled differences) and cutoff value for uniform interval
    fid_max = np.max(np.abs((compare_sample - CI_center) / CI_scale), axis=1, na_rm=True)
    cut_off = np.quantile(fid_max, 1 - alpha, na_rm=True)

    return {
        'mean': CI_center,
        'uniform_lower': CI_center - cut_off * CI_scale,
        'uniform_upper': CI_center + cut_off * CI_scale,
        'median': CI_center,
        'point_lower': np.apply_along_axis(np.quantile, 1, compare_sample, alpha / 2, na_rm=True),
        'point_upper': np.apply_along_axis(np.quantile, 1, compare_sample, 1 - alpha / 2, na_rm=True)
    }

def fid_AUC(fid_sample_top, fid_sample_bottom):
    """
    Function to calculate the Area Under the Curve (AUC) for fiducial samples.

    Input:
    - fid_sample_top: dictionary containing fiducial sample data for the top particle with keys:
        - 'data': dorted data values,
        - 'survival': survival function values for each fiducial sample,
        - 'nfid': number of fiducial samples.
    - fid_sample_bottom: Dictionary containing fiducial sample data for the bottom particle with the same keys as fid_sample_top.

    Output:
    - Dictionary with the following keys:
        - 'AUC': array of AUC values for each fiducial sample,
        - 'nfid': number of fiducial samples.
    """

    nfid = fid_sample_top['nfid']

    # Combine and sort unique data values from both top and bottom samples
    fullgrid = np.sort(np.unique(np.concatenate((fid_sample_top['data'], fid_sample_bottom['data']))))

    # Compute survival functions for the combined grid
    top_surv = particle_grid(fullgrid, fid_sample_top)['survival']
    bottom_surv = particle_grid(fullgrid, fid_sample_bottom)['survival']

    # Initialize array to store AUC values
    auc = np.zeros((nfid, 2))

    # Determine auc values
    for i in range(nfid):
        j = (i % fid_sample_bottom['nfid'])
        auc[i, 0] = 1 + np.sum(np.diff(np.concatenate(([1], top_surv[:, i], [0]))) *
                               (np.concatenate(([1], bottom_surv[:, j])) + np.concatenate(
                                   (bottom_surv[:, j], [0]))) / 2)
        auc[i, 1] = -np.sum(np.diff(np.concatenate(([1], bottom_surv[:, j], [0]))) *
                            (np.concatenate(([1], top_surv[:, i])) + np.concatenate((top_surv[:, i], [0]))) / 2)

    return {'AUC': np.mean(auc, axis=1), 'nfid': nfid}

def calibrationNumber(CI_NP):
    """
    Function to determine fiducial metric 1: average of medians.

    Input:
    - CI_NP: dictionary containing confidence interval data with the key 'median', which holds
    the median values of the confidence intervals.

    Output:
    - Value of average of absolute value medians
    """

    # Identify the index of the last non-NaN median value
    ishow = max(np.where(~np.isnan(CI_NP['median']))[0])

    # Sum over medians and determine average of absolute values
    sum = 0
    for i in CI_NP['median'][0:ishow+1]:
        sum += np.abs(i)
    calib = sum/(ishow+1)

    return calib

def calibrationNumber2(CI_NP):
    """
    Function to determine fiducial metric 2: average of medians scaled by widths of intervals.

    Input:
    - CI_NP: dictionary containing confidence interval data with the key 'median', which holds
    the median values of the confidence intervals.

    Output:
    - Scaled value of average of absolute value medians
    """

    # Identify the index of the last non-NaN median value
    ishow = max(np.where(~np.isnan(CI_NP['median']))[0])

    # Sum over scaled medians and determine average
    sum = 0
    for i in range(ishow+1):
        sum += np.abs(CI_NP['median'][i]) * (CI_NP['point_upper'][i] - CI_NP['point_lower'][i])
    calib = sum/(ishow+1)

    return calib

def calibrationNumber3(CI_NP):
    """
    Function to determine fiducial metric 3: frequency of 0 falling outside of the confidence interval.

    Input:
    - CI_NP: dictionary containing confidence interval data with the key 'median', which holds
    the median values of the confidence intervals.

    Output:
    - Frequency of zero falling outside of the confidence interval.
    """

    # Identify the index of the last non-NaN median value
    ishow = max(np.where(~np.isnan(CI_NP['median']))[0])

    # Count average amount of times that interval contains zero
    sum = 0
    for i in range(ishow+1):
        if CI_NP['point_upper'][i] >= 0 >= CI_NP['point_lower'][i]:
            sum += 1
    calib = sum/(ishow+1)

    return 1-calib

def LRtestNP(data, hlable=['P', 'D'], nfid=1000, ncores=1, GPDgrid=[], display_plot=False, AUC=False):
    """
    Function to perform a likelihood ratio test and analyze results using fiducial inference.

    This function processes LR data for two hypotheses, performs fiducial sampling, calculates survival functions,
    and optionally computes the Area Under the Curve (AUC). It can also generate plots to visualize the results.

    Input:
    - data: dataFrame containing columns 'LLR' (log-likelihood ratios) and 'labels' (P for H_p, D for H_d),
    - hlable: list of two hypothesis labels to distinguish between the two groups in the data,
    - nfid: number of fiducial samples to generate for each hypothesis,
    - ncores: number of cores for parallel processing (not used in the provided code),
    - GPDgrid: custom grid for generating the particle grid. If empty, a default grid is used,
    - display_plot: boolean indicating whether to display diagnostic plots,
    - AUC: Boolean indicating whether to compute and return the Area Under the Curve (AUC).

    Output:
    - A dictionary containing fiducial samples, AUC values (if computed), calibration metrics, and other relevant information.
    """

    # Drop NAN values from data
    data = data.dropna()

    # Create arrays for (log) LR data H_p and H_d
    log_topdata = np.sort(data['LLR'][data['labels'] == hlable[0]])
    topdata = 10 ** log_topdata
    log_bottomdata = np.sort(data['LLR'][data['labels'] == hlable[1]])
    bottomdata = 10 ** log_bottomdata

    # Generate pregrid from min value of log_topdata to maxvalue of log_topdata, stepsize = 1
    if len(GPDgrid) == 0:
        pregrid = np.power(10, np.arange(np.floor(max(-2, min(log_topdata))), np.ceil(min(10, max(log_topdata))) + 1, 1))
    else:
        pregrid = np.power(10, GPDgrid)

    # Make sure bottomdata does not exceed certain threshold
    bottomdata = np.minimum(bottomdata, 2 * max(pregrid))

    # Define the grid and its indices in the pregrid
    grid = np.sort(np.union1d(pregrid, pregrid))
    idgrid = np.array([np.where(x == grid)[0][0] for x in pregrid])

    # Plot the density
    if display_plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        dtop = gaussian_kde(np.log10(topdata))
        dbottom = gaussian_kde(np.log10(bottomdata))
        x_range = np.linspace(min(log_topdata), max(log_bottomdata), 100)
        plt.plot(x_range, dtop(x_range), color='red')
        plt.plot(x_range, dbottom(x_range), color='blue')
        plt.xlabel('log(LR)')
        plt.ylabel('density')
        plt.title('Density of log LR')

        plt.subplot(2, 2, 2)
        plt.plot(np.sort(np.log10(topdata)), 1 - (np.arange(1, len(topdata) + 1) / len(topdata)), color='red')
        plt.plot(np.sort(np.log10(bottomdata)), 1 - (np.arange(1, len(bottomdata) + 1) / len(bottomdata)), color='blue')
        plt.xlabel('log(reported LR)')
        plt.ylabel('probability')
        plt.title('Survival function of log LR')

    # Make a fiducial sample for each hypothesis
    # Generates dictionary with keys: data, u (nfid arrays of length len(data)), len(data), nfid
    fid_sample_top = fiducial_sample(topdata, nfid)
    fid_sample_bottom = fiducial_sample(bottomdata, nfid)

    # Create particle grid and compute survival functions and integrals
    fid_sample_top_grid = particle_grid(grid, fid_sample_top)
    fid_sample_bottom_grid = particle_grid(grid, fid_sample_bottom)

    # Compute and plot AUC
    if AUC != None:
        fid_sample_auc = fid_AUC(fid_sample_top, fid_sample_bottom)
        if display_plot:
            plt.subplot(2, 2, 3)
            plt.boxplot(fid_sample_auc['AUC'])
            plt.ylabel('AUC')
            plt.title('Fiducial distribution of AUC')

    # Compute non-parametric fiducial differences and confidence intervals
    fid_diff_NP = fid_diff_log(fid_sample_top_grid, fid_sample_bottom_grid, idgrid)
    fid_CI_NP = fid_diff_CI(fid_diff_NP)

    # Plot calibration diagnostics
    if display_plot:
        plt.subplot(2, 2, 4)
        dgrid = np.log10(fid_CI_NP['dgrid'])
        plt.plot(dgrid, np.zeros_like(dgrid), color='red', linestyle='--')
        plt.plot(dgrid, fid_CI_NP['median'], color='blue')
        plt.plot(dgrid, fid_CI_NP['uniform_lower'], color='cyan', linestyle='--')
        plt.plot(dgrid, fid_CI_NP['uniform_upper'], color='cyan', linestyle='--')
        plt.plot(dgrid, fid_CI_NP['point_lower'], color='black')
        plt.plot(dgrid, fid_CI_NP['point_upper'], color='black')
        plt.xlabel('log10(reported LR)')
        plt.ylabel('interval-specific calibration discrepancy')
        plt.title('Calibration Diagnostic Plot')
        plt.show()

    # Calculate metric values
    calib = calibrationNumber(fid_CI_NP)
    calib2 = calibrationNumber2(fid_CI_NP)
    calib3 = calibrationNumber3(fid_CI_NP)


    if AUC != None:
        return {'top': fid_sample_top_grid, 'bottom': fid_sample_bottom_grid, 'AUC': fid_sample_auc['AUC'],
                'CI_NP': fid_CI_NP, 'calib': calib, 'calib2': calib2, 'calib3': calib3}
    else:
        return {'top': fid_sample_top_grid, 'bottom': fid_sample_bottom_grid, 'CI_NP': fid_CI_NP,
                'calib': calib, 'calib2': calib2, 'calib3': calib3}

def calibrationPlot(CI_NP, my_title="Calibration Diagnostic Plot", yaxis=None):
    """
    Function to create a calibration diagnostic plot.

    Input:
    - CI_NP: dictionary containing the calibration information with keys:
        'median': median of calibration discrepancies,
        'uniform_lower': lower bound of uniform confidence intervals,
        'uniform_upper': upper bound of uniform confidence intervals,
        'point_lower': lower bound of pointwise confidence intervals,
        'point_upper': upper bound of pointwise confidence intervals,
        'dgrid': the grid of log10(reported LR) values.
    - my_title: Title of the plot (default: "Calibration Diagnostic Plot").
    - yaxis: tuple specifying the y-axis limits; if None, limits are calculated automatically.

    Output:
    - Displays the calibration diagnostic plot.
    """

    # Determine the range of valid indices (non-NaN) for plotting
    ishow = max(np.where(~np.isnan(CI_NP['median']))[0])

    # Set y-limit if not inputted
    if yaxis == None:
        yaxis = [np.floor(min(0, np.nanmin(CI_NP['point_lower']))),
                 np.ceil(max(0, np.nanmax(CI_NP['point_upper'])))]

    # Convert grid values to log scale for x-axis plotting
    dgrid_const = np.log10(CI_NP['dgrid'][:ishow + 2])
    dgrid_const2 = np.sort(np.concatenate((dgrid_const[:-1], dgrid_const[1:])))

    # Median calibration discrepancy
    result_1 = np.repeat(CI_NP['median'][0:ishow + 1], 2)

    # Uniform confidence interval bounds
    result_2 = np.repeat(CI_NP['uniform_lower'][0:ishow+1], 2)
    result_3 = np.repeat(CI_NP['uniform_upper'][0:ishow+1], 2)

    # Pointwise confidence interval bounds
    result_4 = np.repeat(CI_NP['point_lower'][0:ishow+1], 2)
    result_5 = np.repeat(CI_NP['point_upper'][0:ishow+1], 2)


    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot([min(dgrid_const), max(dgrid_const)], [0, 0], color='red', linestyle='--')
    plt.plot(dgrid_const2, result_1, color="blue")
    plt.plot(dgrid_const2, result_2, color="cyan", linestyle="--")
    plt.plot(dgrid_const2, result_3, color="cyan", linestyle="--")
    plt.plot(dgrid_const2, result_4, color="black")
    plt.plot(dgrid_const2, result_5, color="black")
    plt.xlabel('log10(reported LR)')
    plt.ylabel('interval-specific calibration discrepancy')
    plt.title(my_title)
    plt.show()

# FUNCTIONS TO DETERMINE OVERLAP
def overlap(array):
    """
    Function that determines (average) overlap percentage between one array with one or several others.

    Input:
    - array: array of arrays between which the overlap should be determined.

    Output:
    - Average overlap percentage
    """

    # Sort values of first array
    first_vals = np.sort([x for x in array[0] if x is not None])
    number = len(array)

    # Determine percentiles of first array to get 90%-confidence interval
    perc_first_95 = np.percentile(first_vals, 95)
    perc_first_5 = np.percentile(first_vals, 5)

    # Determine the values of the array and the range
    first_within = first_vals[(first_vals <= perc_first_95) & (first_vals >= perc_first_5)]
    ranges = (first_within[0], first_within[-1])
    number_vals = len(first_within)

    # Initialize overlapping values at zero
    overlaps = 0

    # Loop over other arrays and determine overlap percentage with first array
    for i in range(1, number):
        vals = np.sort([x for x in array[i] if x is not None])
        percentage_95 = np.percentile(vals, 95)
        percentage_5 = np.percentile(vals, 5)
        vals_within = vals[(vals <= percentage_95) & (vals >= percentage_5)]
        ranges_vals = (vals_within[0], vals_within[-1])
        number_vals2 = len(vals_within)
        count_within_range_1 = np.sum((vals_within >= ranges[0]) & (vals_within <= ranges[1]))
        count_within_range_2 = np.sum((first_within >= ranges_vals[0]) & (first_within <= ranges_vals[1]))

        # Turn into percentage
        overlap_percentage = min(count_within_range_1,count_within_range_2) / min(number_vals, number_vals2) * 100
        overlaps += overlap_percentage

    # Determine average of overlap percentage
    overlaps = overlaps / (number - 1)

    return overlaps


def average_overlap(array):
    """
    Function that determines average pairwise overlap percentage between several arrays.

    Input:
    - array: array of arrays between which the overlap should be determined.

    Output:
    - Average overlap percentage
    """

    # Initialize values
    number = len(array)
    total_overlap = 0
    pair_count = 0

    # Range over arrays and determine the percentiles
    for i in range(number):
        perfect_vals = np.sort([x for x in array[i] if x is not None])
        perc_perf_95 = np.percentile(perfect_vals, 95)
        perc_perf_5 = np.percentile(perfect_vals, 5)
        perf_within = perfect_vals[(perfect_vals <= perc_perf_95) & (perfect_vals >= perc_perf_5)]
        ranges = (perf_within[0], perf_within[-1])
        number_vals = len(perf_within)

        # Range over leftover arrays and determine overlap
        for j in range(i + 1, number):
            vals = np.sort([x for x in array[j] if x is not None])
            percentage_95 = np.percentile(vals, 95)
            percentage_5 = np.percentile(vals, 5)
            vals_within = vals[(vals <= percentage_95) & (vals >= percentage_5)]
            ranges_vals = (vals_within[0], vals_within[-1])
            number_vals2 = len(vals_within)
            count_within_range_1 = np.sum((vals_within >= ranges[0]) & (vals_within <= ranges[1]))
            count_within_range_2 = np.sum((perf_within >= ranges_vals[0]) & (perf_within <= ranges_vals[1]))
            overlap_percentage = min(count_within_range_1, count_within_range_2) / min(number_vals, number_vals2) * 100
            total_overlap += overlap_percentage
            pair_count += 1

    # Determine average overlap
    average_overlap = total_overlap / pair_count if pair_count > 0 else 0

    return average_overlap

# FUNCTIONS FOR SECOND PART OF RESULTS: GENERATING NEW LR-SYSTEMS

def LSS_calculator(LLRs, probabilities):
    """
    Function that determines frequencies of SS LLRs values based on frequencies of DS-LLRs so that the LR of the LR is the LR.

    Input:
    - LLRs: array of LLR-values (assuming log10 LR).
    - probabiities: array of frequencies with which the LLRs occur for DS.

    Output:
    - Array of same-source LLR-values.
    """

    # Initialize LSS array
    new_LSS = []
    num_bins = len(LLRs)

    # Range over LLRs and generate corresponding SS frequency
    for i in range(num_bins):
        # Determine LR from LLR
        LR = 10**LLRs[i]
        prob_hd = probabilities[i]
        # Generate SS frequency of given LR
        prob_hp = LR * prob_hd
        new_LSS.append(prob_hp)

    return new_LSS

def frequency_creator(data_DS):
    """
    Function that generates a consistent LR-system based on DS LRs and frequencies, so that the LR of the LR is the
    LR and both the SS and DS frequencies sum up to 100.

    Input:
    - data_DS: array of DS LLR-values

    Output:
    - array containing LDS frequencies, LSS frequencies and the corresponding LR values
    """

    # Initialize alpha
    alpha = 1

    # Create array of LLR-values, determine kde of DS-values and using kde, determine frequencies of LLR values
    x_values = np.linspace(min(data_DS), max(data_DS), 10000)
    kde_original = gaussian_kde(data_DS, bw_method='scott')
    LDS_frequencies = kde_original(x_values)

    # While the integrals of the SS and the DS LLRs are not almost equal, loop
    while True:
        # Stretch the LLRs by a factor alpha
        shifted_LLRs = x_values - min(data_DS)
        stretched_shifted = shifted_LLRs * alpha
        stretched_LLRs = stretched_shifted + min(data_DS)

        # Determine LSS frequencies using the LDS frequencies and the stretched LRs
        LSS_frequencies = LSS_calculator(stretched_LLRs, LDS_frequencies)

        # Integrate the two frequencies
        integral_kde = simps(LDS_frequencies, stretched_LLRs)
        integral_LSS = simps(LSS_frequencies, stretched_LLRs)

        # If they are almost equal, done
        if np.isclose(integral_kde, integral_LSS, rtol=0.01, atol=0.01):
            break
        # If there are too many LSS values, decrease alpha
        elif integral_kde < integral_LSS:
            alpha -= 0.001
        # If there are too many LDS values, increase alpha
        else:
            alpha += 0.001

    # Normalize
    LSS_frequencies = LSS_frequencies / np.sum(LSS_frequencies)
    LDS_frequencies = LDS_frequencies / np.sum(LDS_frequencies)

    return [LDS_frequencies, LSS_frequencies, stretched_LLRs]

def calculate_metrics_all(data_SS, data_DS):
    """
    Function that calculates all optimized metrics for SS and DS data.

    Input:
    - data_SS: array of SS LR-values.
    - data_DS: array of DS LR-valies

    Output:
    - values of devPAV, cllr and Fid
    """

    # Initialize metric values to prevent error
    dp = None
    c = None
    cal = None

    # Make arrays of LRs and hypotheses
    lrs = np.concatenate((data_SS, data_DS))
    all_hypotheses = np.concatenate((np.array(['H1'] * len(data_SS)), np.array(['H2'] * len(data_DS))))
    all_hypotheses_01 = np.where(all_hypotheses == 'H1', 1, 0)

    # Determine the metric values
    try:
        dp = devpav_new(lrs, all_hypotheses_01)
    except Exception as e:
        print(f"A devPAV error occurred: {e}")
    try:
        c = cllr(lrs, all_hypotheses_01)
    except Exception as e:
        print(f"A Cllr error occurred: {e}")
    try:
        cal = LRtestNP(pd.DataFrame({'LLR': np.log10(lrs),
                                       'labels': ['P'] * len(data_SS) + ['D'] * len(data_DS)}),
                         nfid=100, AUC=True)['calib'][0]
    except Exception as e:
        print(f"A Fid error occurred: {e}")

    return [c, dp, cal]