# Functions and classes for polarization analyses

## Imports

from numpy import zeros, complex128, exp, pi, real, amax, amin, argmax, abs, array, conj, mean, arctan2, sqrt, arctan, arccos, arctan2, sin, cos, arccos, arcsin, arctan2, row_stack, column_stack, radians
from numpy.linalg import eigh, norm
from scipy.signal import hilbert
from numpy import mean, cov, stack
from matplotlib.pyplot import subplots, Normalize, cm, colorbar
from matplotlib import ticker
import numpy as np
import numpy as np

## Classes
class PolParams:
    def __init__(self, coordinate, window_length, strike, dip, ellipticity, strength, planarity):
        self.coordinate = coordinate
        self.window_length = window_length
        self.strike = strike
        self.dip = dip
        self.ellipticity = ellipticity
        self.strength = strength
        self.planarity = planarity

    def __str__(self):
        return f"Coordinate system: {self.coordinate}\n Strike: {self.strike}\nDip: {self.dip}\nEllipticity: {self.ellipticity}\nStrength: {self.strength}\nPlanarity: {self.planarity}"
    
    def __repr__(self):
        return f"Coordinate system: {self.coordinate}\n Strike: {self.strike}\nDip: {self.dip}\nEllipticity: {self.ellipticity}\nStrength: {self.strength}\nPlanarity: {self.planarity}"

## Functions

### Get the polarization paramters of the the 3C data using the Vidale 1986 method
### Vertical component is positive UPWARD!
def get_pol_vidale(data1, data2, data3, coord="ENZ", window_length=3):

    #### Determine if the data vectors are numpy arrays
    if type(data1) != type(zeros(1)) or type(data2) != type(zeros(1)) or type(data3) != type(zeros(1)):
        raise TypeError("Data vectors must be numpy arrays!")

    #### Determine if the data vectors are the same length
    if len(data1) != len(data2) or len(data1) != len(data3):
        raise ValueError("Data vectors must be the same length!")

    #### Determine if the data are in ENZ or RTZ mode
    if coord != "ENZ" and coord != "RTZ":
        raise ValueError("Mode must be either 'ENZ' or 'RTZ'!")
    
    #### Compute the covariance matrices of the analytic signals of the 3C data
    cov_mats = get_cov_mats(data1, data2, data3, window_length=window_length)

    #### Compute the eigenvalues and eigenvectors of the covariance matrices
    eigvals, eigvecs = get_eigs(cov_mats)

    #### Rotate the eigenvectors to maximize the real components of the first element
    eigvecs_max = eigvecs[:, 2, :] # Eigenvector associated with the largest eigenvalue
    eigvecs_max = rotate_to_maxreal(eigvecs_max)

    #### Get the polarization parameters
    strikes, dips, ellis, strengths, planars = get_pol_from_eigs(eigvals, eigvecs_max)

    #### Create a PolParams object
    pol_dict = {"coordinate": coord, "window_length": window_length, "strike": strikes, "dip": dips, "ellipticity": ellis, "strength": strengths, "planarity": planars}
    pol_params = PolParams(**pol_dict)

    return pol_params

### Get the covariance matrices of the the analytic signals of the 3C data
def get_cov_mats(data1, data2, data3, window_length=3):
        
        #### Determine if the data vectors are the same length
        if len(data1) != len(data2) or len(data1) != len(data3):
            raise ValueError("Data vectors must be the same length!")
        
        #### Compute the covariance matrix of each data point
        data1 = hilbert(data1)
        data2 = hilbert(data2)
        data3 = hilbert(data3)

        numpts = len(data1)-window_length+1
        cov_mats = zeros((3, 3, numpts), dtype=complex128)

        for i in range(numpts):
            x = stack((data1[i:i+window_length], data2[i:i+window_length], data3[i:i+window_length]))
            cov_mats[:, :, i] = cov(x)

        return cov_mats

### Compute the eigenvalues and eigenvectors of the covariance matrices
def get_eigs(cov_mats):
        
    numpts = len(cov_mats[0, 0, :])
    eigvals = zeros((3, numpts))
    eigvecs = zeros((3, 3, numpts), dtype=complex128)

    for i in range(numpts):
        eigvals[:, i], eigvecs[:, :, i] = eigh(cov_mats[:, :, i])

    return eigvals, eigvecs

### Rotate the eigenvectors to maximize the real components of the first element
def rotate_to_maxreal(eigvecs_in):
    
    eigvecs_out = zeros(eigvecs_in.shape, dtype=complex128)

    #### Determine the rotation angle
    numpts = eigvecs_in.shape[1]
    for ipt in range(numpts):
        eigvec = eigvecs_in[:, ipt]
        iang_max = 0
        maxreal = norm(real(eigvec))
        for iang in range(180):
            eigvec_rot = eigvec*exp(1j*iang*pi/180)
            if norm(real(eigvec_rot)) > maxreal:
                iang_max = iang
                maxreal = norm(real(eigvec_rot))
        
        eigvecs_out[:, ipt] = eigvec*exp(1j*iang_max*pi/180)

    return eigvecs_out

### Get the polarization parameters from the eigenvectors in the ENZ coordinate system
### Vertical component is positive UPWARD!
def get_pol_from_eigs(eigvals, eigvecs_max):
    
    npts = len(eigvals[0, :])

    #### Strikes, dips, and ellipticities associated with the largest eigenvalue
    strikes = zeros(npts) # Strike angle is measured clockwise from north/transverse and is in the range [0, 180]
    dips = zeros(npts) # Dip angle is measured downward from horizontal and is in the range [-90, 90]
    ellis = zeros(npts) # 1 means circular, 0 means linear

    #### Polarization strength and the degree of planarity
    strengths = zeros(npts)
    planars = zeros(npts)

    #### Compute the polarization parameters
    eigvals_max = eigvals[2, :]
    eigvals_mid = eigvals[1, :]
    eigvals_min = eigvals[0, :]

    ##### Strikes
    strikes = get_strikes(eigvecs_max)

    ##### Dips
    dips = get_dips(eigvecs_max)

    ##### Ellipticities
    realnorms = norm(real(eigvecs_max), axis=0)
    realnorms[realnorms > 1] = 1 # This is to avoid numerical errors
    ellis = sqrt(1-realnorms**2)/realnorms

    ##### Polarization strength
    strengths = 1-(eigvals_mid+eigvals_min)/eigvals_max

    ##### Degree of planarity
    planars = 1-eigvals_min/eigvals_max

    return strikes, dips, ellis, strengths, planars

### Get the strike angles of the real part of the eigenvectors of all data points
def get_strikes(eigvecs):
            
        #### Determine if the eigenvector is a numpy array
        if type(eigvecs) != type(zeros(1)):
            raise TypeError("The matrix containing the eigenvectors must be a numpy array!")
    
        #### Determine if the array containing the eigenvectors has the correct shape
        if eigvecs.shape[0] != 3:
            raise ValueError("The matrix containing the eigenvectors must have shape (3, npts)!")
    
        #### Compute the strike angle
        eigvecs = real(eigvecs)
        strikes = arctan2(eigvecs[0, :], eigvecs[1, :])*180/pi

        #### Convert the strike angles to the range [0, 180]
        strikes[strikes < 0] += 180
        
        return strikes

### Get the dip angles of the real part of the eigenvectors of all data points
def get_dips(eigvecs):
        
    #### Determine if the eigenvector is a numpy array
    if type(eigvecs) != type(zeros(1)):
        raise TypeError("The matrix containing the eigenvectors must be a numpy array!")

    #### Determine if the array containing the eigenvectors has the correct shape
    if eigvecs.shape[0] != 3:
        raise ValueError("The matrix containing the eigenvectors must have shape (3, npts)!")
    
    #### Reverse the sign of the eigenvector if the first element is negative
    eigvecs = real(eigvecs)
    for i in range(eigvecs.shape[1]):
        if eigvecs[0, i] < 0:
            eigvecs[:, i] *= -1

    #### Compute the dip angle
    dips = arctan2(-eigvecs[2, :], sqrt(eigvecs[0, :]**2 + eigvecs[1, :]**2))*180/pi
    
    return dips

### Plot the 3C waveforms and the polarization parameters as functions of time
def plot_waveforms_and_pols(data1, data2, data3, pol_params, timeax, mode="ENZ", ampmax=1.05, window_length=3, station="A01"):
        
        #### Determine if the data vectors are the same length
        if len(data1) != len(data2) or len(data1) != len(data3):
            raise ValueError("Data vectors must be the same length!")
    
        #### Determine if the data are in ENZ or RTZ mode
        if mode != "ENZ" and mode != "RTZ":
            raise ValueError("Mode must be either 'ENZ' or 'RTZ'!")
        
        #### Normalize the data vectors
        amp = amax([amax(abs(data1)), amax(abs(data2)), amax(abs(data3))])
        data1 = data1 / amp
        data2 = data2 / amp
        data3 = data3 / amp

        #### Define the axes
        fig, axes = subplots(4, 1, figsize=(10, 10), sharex=True)

        #### Plot the waveforms in the first subplot
        timeax_wave = timeax

        if mode == "ENZ":
            ax_wave = axes[0]
            ax_wave.plot(timeax_wave, data1, "royalblue", label="East")
            ax_wave.plot(timeax_wave, data2, "forestgreen", label="North")
            ax_wave.plot(timeax_wave, data3, "black", label="Up")
            ax_wave.set_ylabel("Amplitude")
            ax_wave.legend(loc="upper right")
            ax_wave.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        elif mode == "RTZ":
            ax_wave.plot(timeax_wave, data1, "royalblue", label="Radial")
            ax_wave.plot(timeax_wave, data2, "forestgreen", label="Transverse")
            ax_wave.plot(timeax_wave, data3, "black", label="Vertical")
            ax_wave.set_ylabel("Amplitude")
            ax_wave.legend(loc="upper right")
        
        # # Create a twin Axes object for the top x-axis
        # ax_wave_top = ax_wave.twiny()
        # ax_wave_top.set_xlim(ax_wave.get_xlim())  # Set the same x-axis limits as ax_wave
        
        # # Customize the tick marks and labels on the top x-axis
        # ax_wave_top.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # Set the tick interval to 1.0
        # ax_wave_top.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # Set the minor tick interval to 0.2
        # ax_wave_top.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
        
        ax_wave.set_ylim(-ampmax, ampmax)
        ax_wave.set_title(f"Station {station}", fontsize=16, fontweight="bold")
   
        #### Plot the strikes and dips in the second subplot
        timeax_pol = timeax[window_length-1:] # This is to account for the window length used in the polarization analysis

        ax_strike = axes[1]
        ax_strike.plot(timeax_pol, pol_params.strike, "mediumpurple", label="Strike")
        ax_dip = ax_strike.twinx()
        ax_dip.plot(timeax_pol, pol_params.dip, "teal", label="Dip ($^\circ$)")

        ax_strike.set_ylim(0, 180)
        ax_strike.set_ylabel("Strike ($^\circ$)", color="mediumpurple")

        ax_dip.set_ylim(-90, 90)
        ax_dip.set_ylabel("Dip ($^\circ$)", color="teal")

        ax_strike.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax_strike.spines['left'].set_color('mediumpurple')
        ax_strike.tick_params(axis='y', colors='mediumpurple')
        
        ax_dip.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax_dip.spines['right'].set_color('teal')
        ax_dip.tick_params(axis='y', colors='teal')

        #### Plot the ellipticities in the third subplot
        ax_ellip = axes[2]
        ax_ellip.plot(timeax_pol, pol_params.ellipticity, "black", label="Ellipticity")
        ax_ellip.set_ylim(0, 1)
        ax_ellip.set_ylabel("Ellipticity")
        
        #### Plot the strengths and planarities in the fourth subplot
        ax_stren = axes[3]
        ax_stren.plot(timeax_pol, pol_params.strength, "mediumpurple", label="Strength")
        ax_plan = ax_stren.twinx()
        ax_plan.plot(timeax_pol, pol_params.planarity, "teal", label="Planarity")

        ax_stren.set_ylim(0, 1.05)
        ax_plan.set_ylim(0, 1.05)

        ax_stren.set_ylabel("Strength", color="mediumpurple")
        ax_plan.set_ylabel("Planarity", color="teal")

        ax_stren.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax_stren.spines['left'].set_color('mediumpurple')
        ax_stren.tick_params(axis='y', colors='mediumpurple')

        ax_plan.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax_plan.spines['right'].set_color('teal')
        ax_plan.tick_params(axis='y', colors='teal')

        ax_stren.set_xlim(timeax_wave[0], timeax_wave[-1])
        ax_stren.set_xlabel("Time (s)")

        axdict = {"waveform": ax_wave, "strike": ax_strike, "dip": ax_dip, "ellipticity": ax_ellip, "strength": ax_stren, "planarity": ax_plan}

        return fig, axdict, timeax_pol

### Get the dip angles of the P particle motion from the polarization parameters
def get_p_dip(polparams, timeax_pol, window_length=10, ptime=0.0):
            
        #### Determine if the polarization parameters are a PolParams object
        if type(polparams) != type(PolParams("ENZ", zeros(1), zeros(1), zeros(1), zeros(1), zeros(1))):
            raise TypeError("The polarization parameters must be a PolParams object!")
    
        #### Determine if the time axis is a numpy array
        if type(timeax_pol) != type(zeros(1)):
            raise TypeError("The time axis must be a numpy array!")
        
        #### Determine if the time axis has the correct length
        if len(timeax_pol) != len(polparams.strike):
            raise ValueError("The time axis must have the same length as the polarization parameters!")
        
        #### Extract the P dip
        sampint = timeax_pol[1]-timeax_pol[0]
        ibegin = round((ptime-timeax_pol[0])/sampint)+1
        iend = ibegin+window_length+1

        if iend > len(timeax_pol):
            iend = len(timeax_pol)

        dips_win = polparams.dip[ibegin:iend]

        #### Compute the mean dip
        dip_mean = get_mean_dip(dips_win)
        dip_p = dip_mean

        # maxdip = amax(abs(dips_win))
        # imax = argmax(abs(dips_win))
        # if dips_win[imax] < 0:
        #     maxdip *= -1

        return dip_p

### Get the strike angles of the P and SV particle motion from the polarization parameters
def get_psv_strike(polparams, timeax_pol, window_length=20, ptime=0.0):
        
        #### Determine if the polarization parameters are a PolParams object
        if type(polparams) != type(PolParams("ENZ", zeros(1), zeros(1), zeros(1), zeros(1), zeros(1))):
            raise TypeError("The polarization parameters must be a PolParams object!")
    
        #### Determine if the time axis is a numpy array
        if type(timeax_pol) != type(zeros(1)):
            raise TypeError("The time axis must be a numpy array!")
        
        #### Determine if the time axis has the correct length
        if len(timeax_pol) != len(polparams.strike):
            raise ValueError("The time axis must have the same length as the polarization parameters!")
        
        #### Extract the P and SV strikes
        sampint = timeax_pol[1]-timeax_pol[0]
        ibegin = round((ptime-timeax_pol[0])/sampint)+1
        iend = ibegin+window_length+1

        if iend > len(timeax_pol):
            iend = len(timeax_pol)

        strikes_win = polparams.strike[ibegin:iend]

        #### Compute the mean strike
        strike_mean = get_mean_strike(strikes_win)

        return strike_mean

### Get the mean of a list of dip angles defined in the range [-90, 90]
def get_mean_dip(dips):
    #### Determine if the dips are a numpy array
    if type(dips) != type(zeros(1)):
        raise TypeError("The dips must be a numpy array!")

    #### Determine if the dips have the correct shape
    if len(dips.shape) != 1:
        raise ValueError("The dips must have shape (npts,)!")

    #### Determine if the dips are in the range [-90, 90]
    if amax(dips) > 90 or amin(dips) < -90:
        raise ValueError("The dips must be in the range [-90, 90]!")

    #### Compute the mean dip
    vecs = row_stack((cos(2*radians(dips)), sin(2*radians(dips))))
    vec_mean = mean(vecs, axis=1)
    dip_mean = arctan2(vec_mean[1], vec_mean[0])*90/pi

    #### Convert the mean dip to the range [-90, 90]
    if dip_mean < -90.0:
        dip_mean += 180.0
    elif dip_mean >= 90.0:
        dip_mean -= 180.0

    return dip_mean

### Get the mean of a list of strike angles defined in the range [0, 180]
def get_mean_strike(strikes):
    #### Determine if the strikes are a numpy array
    if type(strikes) != type(zeros(1)):
        raise TypeError("The strikes must be a numpy array!")

    #### Determine if the strikes have the correct shape
    if len(strikes.shape) != 1:
        raise ValueError("The strikes must have shape (npts,)!")

    #### Determine if the strikes are in the range [0, 180]
    if amax(strikes) > 180 or amin(strikes) < 0:
        raise ValueError("The strikes must be in the range [0, 180]!")

    #### Compute the mean strike
    vecs = row_stack((cos(2*radians(strikes)), sin(2*radians(strikes))))
    vec_mean = mean(vecs, axis=1)
    strike_mean = arctan2(vec_mean[1], vec_mean[0])*90/pi

    #### Convert the mean strike to the range [0, 180]
    if strike_mean < 0.0:
        strike_mean += 180.0
    elif strike_mean >= 180.0:
        strike_mean -= 180.0

    return strike_mean

### Plot the strike and dip angles at each station
def plot_station_strike_and_dip(plotdf, source_coord, title, eastmin=-20.0, eastmax=65.0, northmin=-100.0, northmax=-25.0):
        
    #### Determine if the dataframe has the correct columns
    if "station" not in plotdf.columns or "east" not in plotdf.columns or "north" not in plotdf.columns or "strike_psv" not in plotdf.columns or "dip_p" not in plotdf.columns:
        raise ValueError("The dataframe must have columns 'station', 'east', 'north', 'strike_psv', and 'dip_p'!")
        
    #### length of the line segments
    r = 5.0

    #### Offset the station labels
    offset = 2.0

    #### Plot the strike and dip angles
    fig, ax = subplots(1, 1, figsize=(10, 10))
    
    #### Set the background color to light gray
    ax.set_facecolor('lightgray')
    fig.patch.set_alpha(0)

    #### Plot the stations
    ax.plot(plotdf["east"], plotdf["north"], "^", color="white", markersize=15, markeredgewidth=1, markeredgecolor="black")

    #### Plot the source
    ax.plot(source_coord[0], source_coord[1], "*", color="yellow", markersize=15, markeredgewidth=1, markeredgecolor="black")

    #### Map the dip angles to colors
    dips = plotdf["dip_p"].values
    norm = Normalize(amin(dips), amax(dips))
    colors = cm.bwr(norm(dips))
    
    #### Plot short line segments with orientations as strikes and colors as dips
    for i, row in plotdf.iterrows():
        station = row["station"]
        x = row["east"]
        y = row["north"]
        strike = row["strike_psv"]
        dx = r/2*sin(radians(strike))
        dy = r/2*cos(radians(strike))

        ax.plot([x-dx, x+dx], [y-dy, y+dy], color=colors[i], linewidth=2)
        ax.annotate(station, (x+offset, y-offset), fontsize=10)

    ax.set_xlim(eastmin, eastmax)
    ax.set_ylim(northmin, northmax)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_aspect('equal')

    # Add colorbar
    ax_cb = fig.add_axes([0.15, 0.17, 0.02, 0.2])

    cmap = cm.get_cmap('bwr')
    norm = Normalize(vmin=-90, vmax=90)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, cax=ax_cb, orientation='vertical', label='Dip ($^\circ$)')

    # Set the colorbar label spacing to 30
    colorbar.ax.yaxis.set_ticks_position('right')
    colorbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(30))

    return fig, ax

        
            
