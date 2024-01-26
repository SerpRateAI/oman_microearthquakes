# Functions for making animated plots

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# Function to initialize the plot
def init_particle_motion(data_r, data_z, timeax, stname):
    # Create the figure
    fig = plt.figure(figsize=(12, 4))

    # Define the grid layout
    gs = gridspec.GridSpec(1, 3)

    # Define the particle motion axis
    ax_pm = fig.add_subplot(gs[0, 0])
    ax_pm.set_xlim(-1.1, 1.1)
    ax_pm.set_ylim(-1.1, 1.1)
    ax_pm.set_aspect('equal')
    ax_pm.set_xlabel('Radial')
    ax_pm.set_ylabel('Vertical')
    ax_pm.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_pm.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_pm.set_title('Particle Motion', fontsize=12, fontweight='bold')
    ax_pm.text(-1.05, 1.05, stname, fontsize=12, fontweight='bold', va='top', ha='left')
    mappable = ax_pm.scatter(data_r[0], data_z[0], c=[timeax[0]], cmap='jet', vmin=timeax[0], vmax=timeax[-1], zorder=1)

    fig.colorbar(mappable, ax=ax_pm, orientation='vertical', shrink=0.5, label='Time (s)')

    # Define the waveform axis
    ax_wv = fig.add_subplot(gs[0, 1:3])
    ax_wv.plot(timeax, data_r, color='r', linewidth=2, label='Radial')
    ax_wv.plot(timeax, data_z, color='k', linewidth=2, label='Vertical')
    ax_wv.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    dot_r, = ax_wv.plot([], [], 'ro')
    dot_z, = ax_wv.plot([], [], 'ko')
    time_text = ax_wv.text(0.005, 1, "Time = 0.000 s", fontsize=12, va='top', ha='left')

    ax_wv.legend(loc='upper right')
    ax_wv.set_title('Waveform', fontsize=12, fontweight='bold')

    dot_r.set_data(timeax[0], data_r[0])
    dot_z.set_data(timeax[0], data_z[0])

    ax_wv.set_xlim(timeax[0], timeax[-1])
    ax_wv.set_ylim(-1.1, 1.1)
    ax_wv.set_xlabel('Time (s)')
    ax_wv.set_ylabel('Amplitude')

    plt.tight_layout()

    return fig, ax_pm, ax_wv, dot_r, dot_z, time_text

# Function to update the plot in each fraind
def update_particle_motion(fraind, ax_pm, ax_wv, dot_r, dot_z, time_text, data_r, data_z, timeax):

    # Update the particle motion plot
    if fraind > 0:
        ax_pm.plot(data_r[fraind-1:fraind+1], data_z[fraind-1:fraind+1], 'k-', zorder=0)

    ax_pm.scatter(data_r[fraind], data_z[fraind], c=[timeax[fraind]], cmap='jet', vmin=timeax[0], vmax=timeax[-1], zorder=1)

    # Update the waveform plot
    dot_r.set_data(timeax[fraind], data_r[fraind])
    dot_z.set_data(timeax[fraind], data_z[fraind])

    # Update the time text
    time_text.set_text(f"Time: {timeax[fraind]:.3f} s")

    return ax_pm, ax_wv, dot_r, dot_z, time_text