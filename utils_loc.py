from numpy import sqrt, zeros, array, linspace, mean, argmin, unravel_index
from numpy.linalg import norm
from time import time
from obspy import UTCDateTime


# Compute the body-wave travel time between a certain station and every grid point in 3D using a homogeneous velocity model
def get_station_traveltimes_3dgrid_homo(steast, stnorth, vel, numeast = 61, numnorth = 61, numdep = 26, eastmin=-120, eastmax=120, northmin=-120, northmax=120, depmin=0, depmax=50):

    ## Build the meshgrid
    traveltime = zeros((numdep, numnorth, numeast))
    depgrid = linspace(depmin, depmax, num=numdep)
    nogrid = linspace(northmin, northmax, num=numnorth)
    eagrid = linspace(eastmin, eastmax, num=numeast)

    for depind in range(numdep):
        grddep = depgrid[depind]

        for northind in range(numnorth):
            grdnorth = nogrid[northind]

            for eastind in range(numeast):
                grdeast = eagrid[eastind]

                dist = norm(array([steast, stnorth, 0])-array([grdeast, grdnorth, grddep]))
                traveltime[depind, northind, eastind] = dist/vel

    return traveltime, depgrid, nogrid, eagrid

# Compute the surface-wave travel time between a certain station and every grid point in 2D using a homogeneous velocity model
def get_station_traveltimes_2dgrid_homo(steast, stnorth, vel, numeast = 61, numnorth = 61, eastmin=-120, eastmax=120, northmin=-120, northmax=120):
    from numpy import sqrt, zeros, array, linspace
    from numpy.linalg import norm

    ## Build the meshgrid
    traveltime = zeros((numnorth, numeast))
    numgrid = numeast*numnorth
    nogrid = linspace(northmin, northmax, num=numnorth)
    eagrid = linspace(eastmin, eastmax, num=numeast)

    for northind in range(numnorth):
        grdnorth = nogrid[northind]

        for eastind in range(numeast):
            grdeast = eagrid[eastind]

            dist = norm(array([steast, stnorth])-array([grdeast, grdnorth]))
            traveltime[northind, eastind] = dist/vel

    return traveltime, nogrid, eagrid


# Find the event location and origin time through 3D grid search
def locate_event_3dgrid(pickdf, ttdict, depgrid, nogrid, eagrid):


    begin = time()

    numpk = pickdf.shape[0]
    numdep = len(depgrid)
    numnor = len(nogrid)
    numeas = len(eagrid)

    rmsvol = zeros((numdep, numnor, numeas))
    orivol = zeros((numdep, numnor, numeas))

    for depind in range(numdep):
        for norind in range(numnor):
            for easind in range(numeas):
                ### Compute the origin time       
                otimes_sta = zeros(numpk)
                ttimes = zeros(numpk)
                atimes = zeros(numpk)
                for staind, row in pickdf.iterrows():
                    stname = row['station']
                    atime = UTCDateTime(row['time'])

                    ttime = ttdict[stname][depind, norind, easind]
                    otime = atime-ttime
                    otimes_sta[staind] = otime.timestamp
                    ttimes[staind] = ttime
                    atimes[staind] = atime.timestamp

                otime = mean(otimes_sta)
                orivol[depind, norind, easind] = otime
            
                ### Compute the RMS
                rms = 0
                for staind, ttime in enumerate(ttimes):
                    ttime = ttimes[staind]
                    atime = atimes[staind]
                    rms = rms+(atime-otime-ttime)**2

                rms = sqrt(rms/numpk)
                rmsvol[depind, norind, easind] = rms


    ## Find the grid with the smallest RMS
    ind = argmin(rmsvol)
    evdpind, evnoind, eveaind = unravel_index(ind, rmsvol.shape)
    evori = UTCDateTime(orivol[evdpind, evnoind, eveaind])

    ## Compute the predicted arrival times
    atimedict = {}
    for staind, row in pickdf.iterrows():
        stname = row['station']
        atime = UTCDateTime(row['time'])

        ttime = ttdict[stname][evdpind, evnoind, eveaind]
        otime = evori
        atime = otime+ttime
        atimedict[stname] = atime

    end = time()

    end = time()
    print(f"Time elapsed: {end-begin:1f} s.")

    return evdpind, evnoind, eveaind, evori, rmsvol, atimedict

# Plot the RMS distributions
def plot_rms(rmsvol, evdpind, evnoind, eveaind, depgrid, northgrid, eastgrid, stadf, pickdf, evname, rmsmax=0.02):
    import matplotlib.pyplot as plt
    import numpy as np

    eastmin = eastgrid[0]
    eastmax = eastgrid[-1]
    northmin = northgrid[0]
    northmax = northgrid[-1]
    depmax = depgrid[-1]

    evea = eastgrid[eveaind]
    evno = northgrid[evnoind]
    evdp = depgrid[evdpind]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10), gridspec_kw={'width_ratios': [1]})
    ax1.imshow(rmsvol[evdpind, :, :], extent=[eastmin, eastmax, northmin, northmax], origin='lower', vmin=0.0, vmax=rmsmax, aspect='equal')
    
    for _, row in stadf.iterrows():
        stname = row['name']
        east = row['east']
        north = row['north']

        if eastmin < east < eastmax and northmin < north < northmax:
            if stname in pickdf['station'].values:
                ax1.scatter(east, north, color='white', marker='^')
                ax1.text(east, north+0.5, stname, color='white')
            else:
                ax1.scatter(east, north, color='gray', marker='^')

    ax1.scatter(evea, evno, color='red', marker='*')
    ax1.set_ylabel("North (m)")
    ax1.set_title(f"{evname}, RMS at depth {evdp:.0f} m", fontweight='bold')

    imag2 = ax2.imshow(rmsvol[:, evnoind, :], extent=[eastmin, eastmax, depmax, 0], origin='upper', vmin=0.0, vmax=rmsmax, aspect='equal')
    ax2.scatter(evea, evdp, color='red', marker='*')
    ax2.set_title(f"RMS at north {evno:.0f} m", fontweight='bold')

    ax2.set_xlabel("East (m)")
    ax2.set_ylabel("Depth (m)")

    ax3 = fig.add_axes([0.15, 0.3, 0.01, 0.08])
    fig.colorbar(imag2, cax=ax3, orientation='vertical', label='RMS (s)')

    return fig, ax1, ax2, ax3
    #return fig, ax1, ax2
