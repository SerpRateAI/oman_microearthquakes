
from numpy import array, nan, sqrt, std, sum, zeros, delete, mean

# Estimate the standard deviation of a set of samples using the jackknife method
def jackknife_std(samples):
    num_samples = len(samples)
    
    mean_all = mean(samples)
    means_jack = zeros(num_samples)
    for i in range(num_samples):
        samples_jack = delete(samples, i)
        means_jack[i] = mean(samples_jack)

    std_jack = sqrt( (num_samples - 1) / num_samples * sum( (means_jack - mean_all) ** 2 ) )

    return std_jack

# Estimate the covariance matrix of two sets of two sets of samples using the jackknife method
def jackknife_cov(samples1, samples2):
    if len(samples1) != len(samples2):
        raise ValueError("samples1 and samples2 must have the same length")

    num_samples = len(samples1)
    mean_all_1 = mean(samples1)
    mean_all_2 = mean(samples2)
    means_jack_1 = zeros(num_samples)
    means_jack_2 = zeros(num_samples)
    for i in range(num_samples):
        samples1_jack = delete(samples1, i)
        samples2_jack = delete(samples2, i)
        means_jack_1[i] = mean(samples1_jack)
        means_jack_2[i] = mean(samples2_jack)

    cov_jack = (num_samples - 1) / num_samples * sum( (means_jack_1 - mean_all_1) * (means_jack_2 - mean_all_2) )
    var_jack_1 = (num_samples - 1) / num_samples * sum( (means_jack_1 - mean_all_1) ** 2 )
    var_jack_2 = (num_samples - 1) / num_samples * sum( (means_jack_2 - mean_all_2) ** 2 )

    cov_mat = array([[var_jack_1, cov_jack], [cov_jack, var_jack_2]])

    return cov_mat