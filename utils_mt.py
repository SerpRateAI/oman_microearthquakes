######
# Compute the cross-spectrum of a pair of time series using the multitaper method with adaptive weighting
######

### Import necessary modules
from numpy import angle, array, delete, exp, isrealobj, mean, newaxis, arctan2, pi, cos, sin, deg2rad, rad2deg
from numpy import amax, sqrt, sum, tile, var, where, zeros, vstack
from numpy.linalg import norm, inv
from scipy.fft import fft, fftfreq
from scipy.stats import beta, chi2
from scipy.signal import detrend
from scipy.signal.windows import dpss

from matplotlib.pyplot import subplots, plot

from utils_basic import get_angles_diff, get_angles_mean, power2db

TOLERANCE = 1e-3
ALPHA = 0.05 # The significance level for the uncertainties

# Class for storing the results of the auto-spectrum analysis
class MtAutospecParams:
    def __init__(self, **kwargs):
        self.freqax = kwargs.get('freqax', None)
        self.aspec = kwargs.get('aspec', None)
        self.aspec_lo = kwargs.get('aspec_lo', None)
        self.aspec_hi = kwargs.get('aspec_hi', None)

# Class for storing the results of the multitaper cross-spectrum analysis
class MtCspecParams:
    def __init__(self, **kwargs):
        self.freqax = kwargs.get('freqax', None)
        self.aspec1 = kwargs.get('aspec1', None)
        self.aspec2 = kwargs.get('aspec2', None)
        self.trans = kwargs.get('trans', None)
        self.cohe = kwargs.get('cohe', None)
        self.phase_diff = kwargs.get('phase_diff', None)
        self.aspec1_lo = kwargs.get('aspec1_lo', None)
        self.aspec1_hi = kwargs.get('aspec1_hi', None)
        self.aspec2_lo = kwargs.get('aspec2_lo', None)
        self.aspec2_hi = kwargs.get('aspec2_hi', None)
        self.trans_uncer = kwargs.get('trans_uncer', None)
        self.cohe_uncer = kwargs.get('cohe_uncer', None)
        self.phase_diff_uncer = kwargs.get('phase_diff_uncer', None)
        self.phase_diff_jk = kwargs.get('phase_diff_jk', None)

def mt_autospec(signal, taper_mat, ratio_vec, sampling_rate, verbose = True, normalize = False, physical_unit = True):
    """
    Compute the auto-spectrum of a time series using the multitaper method with adaptive weighting

    Parameters
    ----------
    signal : array_like
        The input time series
    taper_mat : array_like
        The taper sequence matrix
    ratio_vec : array_like
        The concentration ratio vector
    sampling_rate : float
        The sampling frequency of the input time series

    Optional Parameters
    -------------------
    get_uncer : bool, optional
        Whether to compute the uncertainties of the auto-spectrum. Default is True.

    verbose : bool, optional
        Whether to print the progress of the computation. Default is True.

    normalize : bool, optional
        Whether to normalize each time series by its standard deviation. Default is False.

    physical_unit : bool, optional
        Whether to return the auto-spectrum in the physical unit (i.e., nm^2 s^-2 Hz^-1 for geophones). Default is True.

    Returns
    -------
    mt_aspec_params : MtAutospecParams
        The results of the multitaper auto-spectrum analysis
    """

    # Verify the input time series are real
    if not isrealobj(signal):
        raise ValueError('The input time series must be real!')
    
    # Verify the taper matrix and the concentration ratio vector have the same number of tapers
    if taper_mat.shape[0] != len(ratio_vec):
        raise ValueError('The taper matrix and the concentration ratio vector must have the same number of tapers!')
    
    ratio_vec = ratio_vec[:, None]
    num_pts = len(signal)

    if verbose:
        print("Preprocessing the input time series...")

    # Detrend the input time series
    signal = detrend(signal)

    # Normalize the input time series
    if normalize:
        signal = signal / sqrt(var(signal))
        physical_unit = False # The auto-spectrum is not physical if the input time series is normalized

    # Compute the tapered eigen time series
    eig_sig = taper_mat * signal

    if verbose:
        print("Computing the eigen-DFTs of the input time series...")

    # Compute the windowed DFTs of the input time series
    # The windowed DFTs are matrices of size (num_taper, num_pts)
    eig_dft_mat = fft(eig_sig, axis=1) 

    # Select only the positive frequencies
    eig_dft_mat = eig_dft_mat[:, :num_pts//2 + 1]

    freqax = fftfreq(num_pts, 1/sampling_rate)
    freqax = freqax[:num_pts//2 + 1]

    num_pts = len(freqax)

    if verbose:
        print("Computing the auto-spectra and the adaptive weights...")
        
    # Compute the eigen-auto-spectra of the input time series
    eig_aspec_mat = abs(eig_dft_mat)**2

    # for i in range(eig_aspec_mat.shape[0]):
    #     print(amax(eig_aspec_mat[i, :]))

    # Normalize the tapered eigen time series by the total power of the tapers
    eig_aspec_mat = eig_aspec_mat / sum(taper_mat ** 2, axis=1, keepdims=True)

    # Compute the adaptive weights following the method of Thomson (1982)
    weight_mat, mt_aspec = get_adapt_weights(signal, eig_aspec_mat, ratio_vec, verbose = verbose)

    # print(amax(mt_aspec))

    # Convert the auto-spectra to the correct scale
    # mt_aspec = mt_aspec / len(signal) # Scale factor is already included when scaled by the total power of the tapers
    mt_aspec[1:-1] *= 2

    # print(amax(power2db(mt_aspec)))
    
    # Compute the uncertainties
    if verbose:
        print("Computing the uncertainties...")

    # Compute the uncertainties of the auto-spectra using the definitions of their statistical properties on P. 370 of Percival and Walden (1993)
    dof = 2 * sum(weight_mat ** 2 * ratio_vec, axis=0) ** 2 / sum(weight_mat ** 4 * ratio_vec ** 2, axis=0)

    mt_aspec_hi = mt_aspec * chi2.ppf(1 - ALPHA / 2, dof) / dof
    mt_aspec_lo = mt_aspec * chi2.ppf(ALPHA / 2, dof) / dof

    # Check the difference between the variance of the input time series and the multitaper estimates
    if verbose:
        print("Checking the validity of the Perseval Theorem for the MT auto-spectrum...")
        check = sum(mt_aspec) / sum(signal ** 2)
        print(f"Power ratio: {check}")

    # Convert the auto-spectra to the physical unit
    if physical_unit:
        mt_aspec /= sampling_rate
        mt_aspec_lo /= sampling_rate
        mt_aspec_hi /= sampling_rate

    # Return the results
    mt_aspec_params = MtAutospecParams(freqax = freqax, aspec = mt_aspec, aspec_lo = mt_aspec_lo, aspec_hi = mt_aspec_hi)

    return mt_aspec_params
    
def mt_cspec(signal1, signal2, taper_mat, ratio_vec, sampling_rate, get_uncer = True, verbose = True, normalize = False, return_jk = False):
    """
    Compute the cross-spectrum of a pair of time series using the multitaper method with adaptive weighting
    The function is modified from mt_cspek_phs.m by J. Collins, R. Sohn, and T. Barreyre,
    which is in turn modified from Alan Chave's Fortran code.

    The sign convention is such that a positive phase difference means signal1 *LEADS* signal2!

    Parameters
    ----------
    signal1 : array_like
        Input time series 1
    signal2 : array_like
        Input time series 2
    taper_mat : array_like
        The taper sequence matrix. The taper sequence is a matrix of size (num_taper, num_pts) generated by scipy.signal.windows.dpss
    ratio_vec : array_like
        The concentration ratio vector. The ratio_vec is a vector of size (num_taper) generated by scipy.signal.windows.dpss
    sampling_rate : float, optional
        The sampling frequency of the input time series. Default is 1000.0 Hz

    Optional Parameters
    -------------------
    get_uncer : bool, optional
        Whether to compute the uncertainties of the cross-spectrum, transfer function, coherence, and phase difference. Default is True.

    verbose : bool, optional
        Whether to print the progress of the computation. Default is True.

    normalize : bool, optional
        Whether to normalize each time series by its standard deviation. Default is False.

    return_jk : bool, optional
        Whether to return the Jackknife phase difference. Default is False.

    Returns
    -------
    mt_cspec_params : MtCspecParams
        The results of the multitaper cross-spectrum analysis
    """

    # Verify the input time series have the same length
    if len(signal1) != len(signal2):
        raise ValueError('The input time series must have the same length!')
    
    # Verify the input time series are real
    if not isrealobj(signal1) or not isrealobj(signal2):
        raise ValueError('The input time series must be real!')
    
    # Verify the taper matrix and the concentration ratio vector have the same number of tapers
    if taper_mat.shape[0] != len(ratio_vec):
        raise ValueError('The taper matrix and the concentration ratio vector must have the same number of tapers!')
    
    ratio_vec = ratio_vec[:, None]

    num_pts = len(signal1)
    num_taper = taper_mat.shape[0]
    
    if verbose:
        print("Preprocessing the input time series...")

    # Detrend the input time series
    signal1 = detrend(signal1)
    signal2 = detrend(signal2)

    # Normalize the input time series
    if normalize:
        signal1 = signal1 / sqrt(var(signal1))
        signal2 = signal2 / sqrt(var(signal2))

    # Compute the tapered eigen time series
    eig_sig1 = taper_mat * signal1
    eig_sig2 = taper_mat * signal2

    if verbose:
        print("Computing the eigen-DFTs of the input time series...")

    # Compute the windowed DFTs of the input time series
    # The windowed DFTs are matrices of size (num_taper, num_pts)
    eig_dft_mat1 = fft(eig_sig1, axis=1) 
    eig_dft_mat2 = fft(eig_sig2, axis=1)

    # Select only the positive frequencies
    eig_dft_mat1 = eig_dft_mat1[:, :num_pts//2]
    eig_dft_mat2 = eig_dft_mat2[:, :num_pts//2]

    freqax = fftfreq(num_pts, 1/sampling_rate)
    freqax = freqax[:num_pts//2]

    num_pts = len(freqax)

    if verbose:
        print("Computing the auto-spectra and the adaptive weights...")

    # Compute the eigen-auto-spectra of the input time series
    eig_aspec_mat1 = abs(eig_dft_mat1)**2
    eig_aspec_mat2 = abs(eig_dft_mat2)**2

    # Compute the adaptive weights following the method of Thomson (1982)
    weight_mat1, mt_aspec1 = get_adapt_weights(signal1, eig_aspec_mat1, ratio_vec, verbose = verbose)
    weight_mat2, mt_aspec2 = get_adapt_weights(signal2, eig_aspec_mat2, ratio_vec, verbose = verbose)

    if verbose:
        print("Computing the cross-spectrum, transfer function, coherence, and phase difference...")

    # Compute the cross-spectrum, transfer function, coherence, and phase difference
    mt_cohe, mt_trans, mt_phase_diff = get_mt_cspec(eig_dft_mat1, eig_dft_mat2)

    # Convert the auto-spectra to the correct scale
    mt_aspec1 = mt_aspec1 / len(signal1)
    mt_aspec2 = mt_aspec2 / len(signal2)

    mt_aspec1[1:-1] *= 2
    mt_aspec2[1:-1] *= 2

    param_dict = {
        "freqax": freqax,
        "aspec1": mt_aspec1,
        "aspec2": mt_aspec2,
        "trans": mt_trans,
        "cohe": mt_cohe,
        "phase_diff": mt_phase_diff,
    }

    # Compute the uncertainties
    if get_uncer:
        if verbose:
            print("Computing the uncertainties...")

        # Compute the uncertainties of the auto-spectra using the definitions of their statistical properties on P. 370 of Percival and Walden (1993)
        dof1 = 2 * sum(weight_mat1 ** 2 * ratio_vec, axis=0) ** 2 / sum(weight_mat1 ** 4 * ratio_vec ** 2, axis=0)
        dof2 = 2 * sum(weight_mat2 ** 2 * ratio_vec, axis=0) ** 2 / sum(weight_mat2 ** 4 * ratio_vec ** 2, axis=0)

        mt_aspec1_hi = mt_aspec1 * chi2.ppf(1 - ALPHA / 2, dof1) / dof1
        mt_aspec1_lo = mt_aspec1 * chi2.ppf(ALPHA / 2, dof1) / dof1

        mt_aspec2_hi = mt_aspec2 * chi2.ppf(1 - ALPHA / 2, dof2) / dof2
        mt_aspec2_lo = mt_aspec2 * chi2.ppf(ALPHA / 2, dof2) / dof2

        # Compute the uncertainties of the coherence, transfer function, and phase difference using jackknife resampling
        mt_trans_jk_mat = zeros((num_taper, num_pts))
        mt_cohe_jk_mat = zeros((num_taper, num_pts)) 
        mt_phase_diff_jks = zeros((num_taper, num_pts))
        for i in range(num_taper):
            # Remove the ith eigen-dft
            eig_dft_mat1_jk = delete(eig_dft_mat1, i, axis=0)
            eig_dft_mat2_jk = delete(eig_dft_mat2, i, axis=0)

            # Compute the cross-spectrum, transfer function, coherence, and phase difference
            mt_cohe_jk, mt_trans_jk, mt_phase_diff_jk = get_mt_cspec(eig_dft_mat1_jk, eig_dft_mat2_jk)

            # Store the results
            mt_trans_jk_mat[i, :] = mt_trans_jk
            mt_cohe_jk_mat[i, :] = mt_cohe_jk
            mt_phase_diff_jks[i, :] = mt_phase_diff_jk

        # Compute the uncertainties for the transfer function and coherence
        mt_trans_jk_mean = mean(mt_trans_jk_mat, axis=0)
        mt_cohe_jk_mean = mean(mt_cohe_jk_mat, axis=0)

        mt_trans_jk_var = (num_taper - 1) / num_taper * sum((mt_trans_jk_mean - mt_trans_jk_mat) ** 2, axis=0)
        mt_cohe_jk_var = (num_taper - 1) / num_taper * sum((mt_cohe_jk_mean - mt_cohe_jk_mat) ** 2, axis=0)

        mt_trans_uncer = sqrt(mt_trans_jk_var)
        mt_cohe_uncer = sqrt(mt_cohe_jk_var)

        # Compute the uncertainties for the phase difference while avoiding the wrap-around problem by using the vector mean
        mt_phase_diff_jk_mean = get_angles_mean(mt_phase_diff_jks, axis=0)

        mt_phase_diff_var = (num_taper - 1) / num_taper * sum(get_angles_diff(mt_phase_diff_jk_mean, mt_phase_diff_jks) ** 2, axis=0)
        mt_phase_diff_uncer = sqrt(mt_phase_diff_var)

        # Convert the auto-spectra uncertainties to the correct scale
        mt_aspec1_lo = mt_aspec1_lo / len(signal1)
        mt_aspec1_hi = mt_aspec1_hi / len(signal1)

        mt_aspec1_lo[1:-1] *= 2
        mt_aspec1_hi[1:-1] *= 2

        param_dict["aspec1_lo"] = mt_aspec1_lo
        param_dict["aspec1_hi"] = mt_aspec1_hi
        param_dict["aspec2_lo"] = mt_aspec2_lo
        param_dict["aspec2_hi"] = mt_aspec2_hi
        param_dict["trans_uncer"] = mt_trans_uncer
        param_dict["cohe_uncer"] = mt_cohe_uncer
        param_dict["phase_diff_uncer"] = mt_phase_diff_uncer

        if return_jk:
            param_dict["phase_diff_jk"] = mt_phase_diff_jks

    # Check the difference between the variance of the input time series and the multitaper estimates
    if verbose:
        print("Checking the difference between the variance of the input time series and the multitaper estimates...")
        check1 = sum(mt_aspec1) / var(signal1)
        check2 = sum(mt_aspec2) / var(signal2)
        print(f"Check 1: {check1}")
        print(f"Check 2: {check2}")

    # Return the results
    cspec_param = MtCspecParams(**param_dict)

    return cspec_param

# Compute the adaptive weights following the method of Thomson (1982)
def get_adapt_weights(signal, eig_aspec_mat, ratio_vec, tol = TOLERANCE, verbose = True):
    """
    Compute the adaptive weights following the method of Thomson (1982)

    Parameters
    ----------
    signal_mat : array_like
        The windowed DFTs of the input time series. The windowed DFTs are matrices of size (num_taper, num_pts)
    signal_psd_mat : array_like
        The power spectra of the input time series. The power spectra are matrices of size (num_taper, num_pts)
    ratio_vec : array_like
        The concentration ratio vector. The ratio_vec is a vector of size (num_taper) generated by scipy.signal.windows.dpss

    Optional Parameters
    -------------------
    tol : float, optional
        The tolerance for the adaptive weights. Default is 1e-3

    Returns
    -------
    weight_mat: array_like
        The adaptive weights in a matrix of size (num_taper, num_pts)
    """

    num_taper = eig_aspec_mat.shape[0]

    # Get the initial spectral estimates
    var_sig = var(signal)

    mt_aspec_init = mean(eig_aspec_mat[:2, :], axis=0)
    mt_aspec_old = mt_aspec_init

    # Perform the iterative calculation
    num_iter = 0
    if verbose:
        print(f"Performing the iterative calculation with a tolerance of {tol}...")

    while True:
        num_iter += 1

        if verbose:
            print(f"Iteration {num_iter}...")

        # Compute the weights
        weight_mat = tile(mt_aspec_old, (num_taper, 1)) / (ratio_vec * tile(mt_aspec_old, (num_taper, 1)) + (1 - ratio_vec) * var_sig)

        # Compute the new spectral estimates
        mt_aspec_new = sum(ratio_vec * weight_mat ** 2 * eig_aspec_mat, axis=0) / sum(ratio_vec * weight_mat ** 2, axis=0)

        # Check convergence
        update = mean(abs(mt_aspec_new - mt_aspec_old) / mt_aspec_old)
        if verbose:
            print(f"Update: {update}")
        if mean(abs(mt_aspec_new - mt_aspec_old) / mt_aspec_old) < tol:
            if verbose:
                print("Converged!")
            break
        else:
            mt_aspec_old = mt_aspec_new

    mt_aspec = mt_aspec_new

    return weight_mat, mt_aspec

# Compute the cross-spectrum, squared transfer function, coherence, and phase difference from the eigen-dft matrices
# Currentlky, the quantities are computed with simple averaging, NOT with adaptive weights!
def get_mt_cspec(eig_dft_mat1, eig_dft_mat2):
    """
    Compute the cross-spectrum from the eigen-dft matrices

    Parameters
    ----------
    eigen_dft_mat1 : array_like
        The eigen-dft matrix of the first time series. The eigen-dft matrix is of size (num_taper, num_pts)
    eigen_dft_mat2 : array_like
        The eigen-dft matrix of the second time series. The eigen-dft matrix is of size (num_taper, num_pts)

    Returns
    -------
    mt_cohe : array_like
        The squared coherence

    mt_trans : array_like
        The absolute value of the transfer function
    """

    # Compute the cross-spectrum
    num_taper = eig_dft_mat1.shape[0]
    mt_cspec = sum(eig_dft_mat1 * eig_dft_mat2.conj(), axis=0) / num_taper
    mt_trans = sum(abs(eig_dft_mat2 / eig_dft_mat1) ** 2, axis=0) / num_taper
    mt_cohe = abs(mt_cspec) ** 2 / (sum(abs(eig_dft_mat1) ** 2, axis=0) / num_taper) / (sum(abs(eig_dft_mat2) ** 2, axis=0) / num_taper)
    mt_phase_diff = angle(mt_cspec)

    return mt_cohe, mt_trans, mt_phase_diff

# Get the average phase difference in a given frequency range
# The phase differences are weighted by their uncertainties before averaging
# Only the phase differences with coherence greater than min_cohe are selected.
# Only the phase differences more than nw bins aparts are treated as independent measurements and  included in the average
def get_avg_phase_diff(freq_range, freqax, phase_diffs, phase_diff_uncers, coherences, min_cohe = 0.95, nw = 3, return_samples = False, verbose = True):
    """
    Get the average phase difference in a given frequency range

    Parameters
    ----------
    freqax : array_like
        The frequency axis
    mt_phase_diff : array_like
        The phase difference
    mt_phase_diff_uncer : array_like
        The uncertainty of the phase differences
    freq_range : tuple
        The frequency range

    Returns
    -------
    avg_phase_diff : float
        The average phase difference in the given frequency range

    avg_phase_diff_uncer : float
        The uncertainty of the average phase difference
    """

    # Select the phase differences in the given frequency range and with coherence greater than min_cohe
    min_freq, max_freq = freq_range
    freq_inds = where((freqax >= min_freq) & (freqax <= max_freq))[0]
    phase_diffs = phase_diffs[freq_inds]
    phase_diff_uncers = phase_diff_uncers[freq_inds]
    coherences = coherences[freq_inds]

    # Select the phase differences with coherence greater than min_cohe
    inds = coherences > min_cohe
    phase_diffs = phase_diffs[inds]
    phase_diff_uncers = phase_diff_uncers[inds]
    coherences = coherences[inds]
    freq_inds = freq_inds[inds]
    if len(phase_diffs) == 0:
        if verbose:
            print("No phase difference meets the criteria!")

        if return_samples:
            return None, None, None
        else:
            return None, None
        
    elif len(phase_diffs) == 1:
        if verbose:
            print("Only one phase difference meets the critieria!")

        if return_samples:
            return phase_diffs[0], phase_diff_uncers[0], freq_inds[0]
        else:
            return phase_diffs[0], phase_diff_uncers[0]

    # Get the independent phase differences
    phase_diffs_indep, phase_diff_uncers_indep, freq_inds_indep = get_indep_phase_diffs(phase_diffs, phase_diff_uncers, freq_inds, nw)

    num_indep = len(phase_diffs_indep)
    if verbose:
        print(f"Number of independent phase differences contributing to the average: {num_indep}")

    # Compute the weighted average of the phase differences and estimate the uncertainty
    if num_indep == 1:
        if return_samples:
            return phase_diffs_indep[0], phase_diff_uncers_indep[0], freq_inds_indep[0]
        else:
            return phase_diffs_indep[0], phase_diff_uncers_indep[0]

    # Compute the weighted average of the phase differences and estimate the uncertainty
    avg_phase_diff_uncer = sqrt( 1 / sum(1 / phase_diff_uncers_indep ** 2) )
    avg_phase_diff = angle(sum(exp(1j * phase_diffs_indep) / phase_diff_uncers_indep ** 2) * avg_phase_diff_uncer ** 2)

    if return_samples:
        return avg_phase_diff, avg_phase_diff_uncer, freq_inds_indep
    else:
        return avg_phase_diff, avg_phase_diff_uncer

# Get the vector mean of a set of phase differences weighted by their uncertainties
def get_angle_vector_mean(phase_diffs, phase_diff_uncers):
    """
    Get the vector mean of a set of phase differences weighted by their uncertainties

    Parameters
    ----------
    phase_diffs : array_like
        The phase differences
    phase_diff_uncers : array_like
        The uncertainties of the phase differences

    Returns
    -------
    vector_mean : float
        The vector mean of the phase differences
    """

    # Compute the weighted sum of the phase differences
    weighted_sum = sum(exp(1j * phase_diffs) / phase_diff_uncers ** 2)

    # Compute the vector mean
    vector_mean = angle(weighted_sum)

    return vector_mean

# Get independent phase differences using the greedy algorithm
# Independent phase differences are defined as phase differences more than nw bins apart
def get_indep_phase_diffs(phase_diffs_in, phase_diff_uncers_in, freq_inds, nw):
    phase_diffs_in = list(phase_diffs_in)
    phase_diff_uncers_in = list(phase_diff_uncers_in)
    freq_inds = list(freq_inds)

    # Compute the distance matrix
    num_freq = len(freq_inds)
    dist_mat = zeros((num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            dist_mat[i, j] = abs(freq_inds[i] - freq_inds[j])

    # Find the phase difference with the least number of neighbors as the starting point
    # Neighbors are defined as phase points that are less than or equal to nw bins apart
    min_neighbors = num_freq
    ind_init = 0
    for ind in range(num_freq):
        neighbors = sum(dist_mat[ind] <= nw)
        if neighbors < min_neighbors:
            min_neighbors = neighbors
            ind_init = ind

    phase_diff_init = phase_diffs_in[ind_init]
    phase_diff_uncer_init = phase_diff_uncers_in[ind_init]
    freq_ind = freq_inds[ind_init]

    phase_diffs_out = [phase_diff_init]
    phase_diff_uncers_out = [phase_diff_uncer_init]
    freq_inds_out = [freq_ind]
    
    phase_diffs_remain = phase_diffs_in.copy()
    phase_diff_uncers_remain = phase_diff_uncers_in.copy()
    freq_inds_remain = freq_inds.copy()

    phase_diffs_remain.remove(phase_diff_init)
    phase_diff_uncers_remain.remove(phase_diff_uncer_init)
    freq_inds_remain.remove(freq_ind)

    # Remove the phase differences and their uncertainties that are less than nw bins apart from the initial phase difference
    for phase_diff in phase_diffs_remain:
        ind = phase_diffs_in.index(phase_diff)
        phase_diff_uncer = phase_diff_uncers_in[ind]
        freq_ind = freq_inds[ind]
        if dist_mat[ind_init, ind] <= nw:
            phase_diffs_remain.remove(phase_diff)
            phase_diff_uncers_remain.remove(phase_diff_uncer)
            freq_inds_remain.remove(freq_ind)

    # Iterate through the remaining phase differences
    while phase_diffs_remain:
        phase_diff = phase_diffs_remain.pop(0) 
        phase_diffs_out.append(phase_diff)

        phase_diff_uncer = phase_diff_uncers_remain.pop(0)
        phase_diff_uncers_out.append(phase_diff_uncer)

        freq_ind = freq_inds_remain.pop(0)
        freq_inds_out.append(freq_ind)
        
        ind1 = phase_diffs_in.index(phase_diff)
        for phase_diff in phase_diffs_remain:
            ind2 = phase_diffs_in.index(phase_diff)
            phase_diff_uncer = phase_diff_uncers_in[ind2]
            freq_ind = freq_inds[ind2]
            if dist_mat[ind1, ind2] <= nw:
                phase_diffs_remain.remove(phase_diff)
                phase_diff_uncers_remain.remove(phase_diff_uncer)
                freq_inds_remain.remove(freq_ind)

    # Convert the lists to arrays
    phase_diffs_out = array(phase_diffs_out)
    phase_diff_uncers_out = array(phase_diff_uncers_out)
    freq_inds_out = array(freq_inds_out)

    return phase_diffs_out, phase_diff_uncers_out, freq_inds_out

# Get frequency indices that are more than nw bins apart using the greedy algorithm
def get_indep_freq_inds(freq_inds, nw):
    """
    Get frequency indices that are more than nw bins apart
    """
    
    # Convert the frequency indices to list if they are not already
    freq_inds = list(freq_inds)

    # Compute the distance matrix
    num_freq = len(freq_inds)
    dist_mat = zeros((num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            dist_mat[i, j] = abs(freq_inds[i] - freq_inds[j])
    
    # Find the frequency index with the least number of neighbors as the starting point
    min_neighbors = num_freq
    ind_init = 0
    for ind in range(num_freq):
        neighbors = sum(dist_mat[ind] <= nw)
        if neighbors < min_neighbors:
            min_neighbors = neighbors
            ind_init = ind

    freq_ind_init = freq_inds[ind_init]

    freq_inds_out = [freq_ind_init]
    freq_inds_remain = freq_inds.copy()
    freq_inds_remain.remove(freq_ind_init)

    # Remove the frequency indices that are less than nw bins apart from the initial frequency index
    for freq_ind in freq_inds_remain:
        ind = freq_inds.index(freq_ind)
        if dist_mat[ind_init, ind] <= nw:
            freq_inds_remain.remove(freq_ind)

    # Iterate through the remaining frequency indices
    while freq_inds_remain:
        freq_ind = freq_inds_remain.pop(0)
        freq_inds_out.append(freq_ind)

    # Convert the list to an array
    freq_inds_out = array(freq_inds_out)

    return freq_inds_out
        


