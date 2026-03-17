from numpy import array, vstack, sin, cos, arctan2, arange, ndarray, zeros, isnan, inf, mean, sqrt
from numpy.linalg import norm, inv
from pathlib import Path
from h5py import File, string_dtype
from typing import Dict, Any
from pandas import Timestamp, Timedelta
from tqdm import tqdm
from pandas import DataFrame

from utils_basic import (
    VEL_MODEL_DIR as dirpath_vel,
    LOC_DIR as dirpath_loc,
    DETECTION_DIR as dirpath_detection,
    get_template_subarray
)

from matplotlib.pyplot import figure, subplots, imshow, colorbar, show, tight_layout, savefig

from utils_plot import format_east_xlabels, format_north_ylabels, format_depth_ylabels

"""
Save the travel time volumes to a HDF5 file.
"""
def save_travel_time_volumes_individual(filepath, phase, subarray, scale_factor, easts_grid, norths_grid, depths_grid, travel_time_dict):

    # Check if the file exists
    with File(filepath, "a") as f:
        f.attrs["phase"] = phase
        f.attrs["subarray"] = subarray
        f.attrs["scale_factor"] = scale_factor    

        # Save the axes
        if "east_grid" in f.keys():
            del f["east_grid"]
        f.create_dataset("east_grid", data = easts_grid, dtype=float, shape=easts_grid.shape)

        if "north_grid" in f.keys():
            del f["north_grid"]
        f.create_dataset("north_grid", data = norths_grid, dtype=float, shape=norths_grid.shape)

        if "depth_grid" in f.keys():
            del f["depth_grid"]
        f.create_dataset("depth_grid", data = depths_grid, dtype=float, shape=depths_grid.shape)

        # Save the stations
        stations = list(travel_time_dict.keys())
        dt = string_dtype(encoding="utf-8")

        # Delete the stations dataset if it exists
        if "stations" in f.keys():
            del f["stations"]

        f.create_dataset("stations",
                                      data=array(stations, dtype=dt),
                                      dtype=dt, shape=len(stations))

        # Save the travel time volumes for each station
        for station, travel_time_vol in travel_time_dict.items():
            station_group = f.require_group(station)

            if "travel_time" in station_group.keys():
                del station_group["travel_time"]

            # Save the travel time volume
            station_group.create_dataset("travel_time", data = travel_time_vol, dtype=float, shape=travel_time_vol.shape)

            print(f"Saved travel time volume of {phase} for station {station} and scale factor {scale_factor} to {filepath}.")

"""
Combine the travel time volumes of multiple scale factors into a single HDF5 file.
"""
def save_travel_time_volumes_combined(filepath, scale_factor, easts_grid, norths_grid, depths_grid, travel_time_dict):

    # Check if the file exists
    with File(filepath, "a") as f:
        # Get the group for the scale factor
        scale_factor_group = f.require_group(f"{scale_factor:.1f}")

        # Save the axes
        if "east_grid" in scale_factor_group.keys():
            del scale_factor_group["east_grid"]
        scale_factor_group.create_dataset("east_grid", data = easts_grid, dtype=float, shape=easts_grid.shape)

        if "north_grid" in scale_factor_group.keys():
            del scale_factor_group["north_grid"]
        scale_factor_group.create_dataset("north_grid", data = norths_grid, dtype=float, shape=norths_grid.shape)

        if "depth_grid" in scale_factor_group.keys():
            del scale_factor_group["depth_grid"]
        scale_factor_group.create_dataset("depth_grid", data = depths_grid, dtype=float, shape=depths_grid.shape)

        # Save the stations
        stations = list(travel_time_dict.keys())
        dt = string_dtype(encoding="utf-8")

        # Delete the stations dataset if it exists
        if "stations" in scale_factor_group.keys():
            del scale_factor_group["stations"]

        scale_factor_group.create_dataset("stations",
                                      data=array(stations, dtype=dt),
                                      dtype=dt, shape=len(stations))

        # Save the travel time volumes for each station
        for station, travel_time_vol in travel_time_dict.items():
            station_group = scale_factor_group.require_group(station)

            if "travel_time" in station_group.keys():
                del station_group["travel_time"]

            # Save the travel time volume
            station_group.create_dataset("travel_time", data = travel_time_vol, dtype=float, shape=travel_time_vol.shape)

            print(f"Saved travel time volumes of {station} for scale factor {scale_factor} to {filepath}.")

"""
Load the travel time volumes of a certain phase and subarray from a HDF5 file.
"""
def load_travel_time_volumes_individual(filepath):

    with File(filepath, "r") as f:
        # Read the axes
        easts_grid = f["east_grid"][:]
        norths_grid = f["north_grid"][:]
        depths_grid = f["depth_grid"][:]

        # Read the stations
        stations = f["stations"][:]
        stations = [station.decode("utf-8") for station in stations]

        # Read the travel time volumes for each station
        travel_time_dict = {}
        for station in stations:
            travel_time_dict[station] = f[station]["travel_time"][:]

        return easts_grid, norths_grid, depths_grid, travel_time_dict
    
"""
Load the travel time volumes of a certain scale factor from a combined HDF5 file.
"""
def load_travel_time_volumes_combined(filepath, scale_factor = 1.0):
    with File(filepath, "r") as f:
        print(f.keys())
        scale_factor_group = f[f"{scale_factor:.1f}"]

        # Read the axes
        easts_grid = scale_factor_group["east_grid"][:]
        norths_grid = scale_factor_group["north_grid"][:]
        depths_grid = scale_factor_group["depth_grid"][:]

        # Read the stations
        stations = scale_factor_group["stations"][:]
        stations = [station.decode("utf-8") for station in stations]

        # Read the travel time volumes for each station
        travel_time_dict = {}
        for station in stations:
            travel_time_dict[station] = scale_factor_group[station]["travel_time"][:]

        return easts_grid, norths_grid, depths_grid, travel_time_dict

"""
Save location information of a grid search to an individual HDF5 file.
The location information includes the origin time, location, minimum RMS, predicted arrival times, and the RMS volume.
"""
def save_location_to_hdf_individual(filepath, event_type, event_id, arrival_type, phase, subarray, scale_factor, weight, location_dict, arrival_time_dict, misfit_dict, easts_grid, norths_grid, depths_grid, misfit_vol):

    with File(filepath, "w") as f:
        # Save the location information
        origin_time = location_dict["origin_time"].value # Convert to nanoseconds since the Unix epoch
        east = location_dict["east"]
        north = location_dict["north"]
        depth = location_dict["depth"]
        misfit = location_dict["min_misfit"]

        # Save the event information
        f.attrs["event_type"] = event_type
        f.attrs["event_id"] = event_id
        f.attrs["arrival_type"] = arrival_type
        f.attrs["phase"] = phase
        f.attrs["scale_factor"] = scale_factor
        f.attrs["subarray"] = subarray
        f.attrs["weight"] = weight

        f.attrs["origin_time"] = origin_time
        f.attrs["east"] = east
        f.attrs["north"] = north
        f.attrs["depth"] = depth
        f.attrs["min_misfit"] = misfit

        # Save the predicted arrival times
        arrival_group = f.require_group("arrival_time")
        for station, arrival_time in arrival_time_dict.items():
            arrival_group.attrs[station] = arrival_time.value # Convert to nanoseconds since the Unix epoch

        # Save the misfit at each station (in seconds)
        misfit_group = f.require_group("misfit")
        for station, misfit in misfit_dict.items():
            misfit_group.attrs[station] = misfit

        # Save the misfit volume
        volume_group = f.require_group("misfit_volume")

        # Save the axes
        if "east_grid" in volume_group.keys():
            del volume_group["east_grid"]
        volume_group.create_dataset("east_grid", data=easts_grid, dtype=float, shape=easts_grid.shape)

        if "north_grid" in volume_group.keys():
            del volume_group["north_grid"]
        volume_group.create_dataset("north_grid", data=norths_grid, dtype=float, shape=norths_grid.shape)

        if "depth_grid" in volume_group.keys():
            del volume_group["depth_grid"]
        volume_group.create_dataset("depth_grid", data=depths_grid, dtype=float, shape=depths_grid.shape)

        # Save the misfit volume
        if "misfit" in volume_group.keys():
            del volume_group["misfit"]
        volume_group.create_dataset("misfit", data=misfit_vol, dtype=float, shape=misfit_vol.shape)

    print(f"Saved the location information to {filepath}.")

"""
Save the location information of multiple grid searches to a combined HDF5 file
"""
def save_location_to_hdf_combined(filepath, param_dict, location_dict, arrival_time_dict, station_misfit_dict, grid_dict):
    # Unpack the parameters
    arrival_type = param_dict["arrival_type"]
    phase = param_dict["phase"]
    subarray = param_dict["subarray"]
    scale_factor = param_dict["scale_factor"]
    weight = param_dict["weight"]

    # Get the grid information
    easts_grid = grid_dict["easts_grid"]
    norths_grid = grid_dict["norths_grid"]
    depths_grid = grid_dict["depths_grid"]
    misfit_vol = grid_dict["misfit_vol"]

    with File(filepath, "a") as f:
        # Create a group for the arrival type
        arrival_type_group = f.require_group(arrival_type)
        phase_group = arrival_type_group.require_group(phase)
        scale_factor_group = phase_group.require_group(f"{scale_factor:.1f}")
        
        # Save the location information
        origin_time = location_dict["origin_time"].value # Convert to nanoseconds since the Unix epoch
        east = location_dict["east"]
        north = location_dict["north"]
        depth = location_dict["depth"]
        misfit = location_dict["min_misfit"]
        
        # Save the location information
        scale_factor_group.attrs["origin_time"] = origin_time
        scale_factor_group.attrs["east"] = east
        scale_factor_group.attrs["north"] = north
        scale_factor_group.attrs["depth"] = depth
        scale_factor_group.attrs["min_misfit"] = misfit

        # Save the weight
        scale_factor_group.attrs["weight"] = weight

        # Save the predicted arrival times
        arrival_group = scale_factor_group.require_group("arrival_time")
        for station, arrival_time in arrival_time_dict.items():
            arrival_group.attrs[station] = arrival_time.value # Convert to nanoseconds since the Unix epoch
        
        # Save the misfit at each station (in seconds)
        misfit_group = scale_factor_group.require_group("station_misfit")
        for station, misfit in station_misfit_dict.items():
            misfit_group.attrs[station] = misfit

        # Save the misfit volume
        volume_group = scale_factor_group.require_group("misfit_volume")

        # Save the axes
        if "east_grid" in volume_group.keys():
            del volume_group["east_grid"]
        volume_group.create_dataset("east_grid", data=easts_grid, dtype=float, shape=easts_grid.shape)

        if "north_grid" in volume_group.keys():
            del volume_group["north_grid"]
        volume_group.create_dataset("north_grid", data=norths_grid, dtype=float, shape=norths_grid.shape)

        if "depth_grid" in volume_group.keys():
            del volume_group["depth_grid"]
        volume_group.create_dataset("depth_grid", data=depths_grid, dtype=float, shape=depths_grid.shape)

        if "misfit" in volume_group.keys():
            del volume_group["misfit"]
        volume_group.create_dataset("misfit", data=misfit_vol, dtype=float, shape=misfit_vol.shape)

    print(f"Saved the location information to {filepath}.")

"""
Read the station misfits of a grid search result from an HDF5 file
"""
def load_station_misfits_from_hdf_combined(filepath, arrival_type, phase, subarray, scale_factor):
    with File(filepath, "r") as f:
        arrival_type_group = f[arrival_type]
        phase_group = arrival_type_group[phase]
        subarray_group = phase_group[subarray]
        print(subarray_group.keys())
        scale_factor_group = subarray_group[f"{scale_factor:.1f}"]
        
        misfit_dict = {}
        misfit_group = scale_factor_group.require_group("station_misfit")
        for station in misfit_group.attrs.keys():
            misfit_dict[station] = misfit_group.attrs[station]

        return misfit_dict

"""
Load the predicted arrival times from a combined HDF5 file
"""
def load_predicted_arrival_times_from_hdf_combined(filepath, arrival_type, phase, scale_factor):

    with File(filepath, "r") as f:
        arrival_type_group = f[arrival_type]
        phase_group = arrival_type_group[phase]
        scale_factor_group = phase_group[f"{scale_factor:.1f}"]
        
        # Load the predicted arrival times
        arrival_time_dict = {}
        arrival_group = scale_factor_group.require_group("arrival_time")
        for station in arrival_group.attrs.keys():
            arrival_time_dict[station] = Timestamp(arrival_group.attrs[station], unit="ns")

        # Load the origin time
        origin_time = Timestamp(scale_factor_group.attrs["origin_time"], unit="ns")

        return arrival_time_dict, origin_time

"""
Load the location information from an HDF5 file.
"""
def load_location_from_hdf_individual(filepath):
    with File(filepath, "r") as f:
        # Load the location information
        event_type = f.attrs["event_type"]
        event_id = f.attrs["event_id"]
        arrival_type = f.attrs["arrival_type"]
        phase = f.attrs["phase"]
        scale_factor = f.attrs["scale_factor"]
        subarray = f.attrs["subarray"]
        weight = f.attrs["weight"]

        origin_time = Timestamp(f.attrs["origin_time"], unit="ns")
        east = f.attrs["east"]
        north = f.attrs["north"]
        depth = f.attrs["depth"]
        misfit = f.attrs["min_misfit"]

        location_dict = {
            "origin_time": origin_time,
            "east": east,
            "north": north,
            "depth": depth,
            "min_misfit": misfit
        }

        # Load the predicted arrival times
        arrival_time_dict = {}
        arrival_group = f["arrival_time"]
        for station in arrival_group.attrs.keys():
            arrival_time_dict[station] = Timestamp(arrival_group.attrs[station], unit="ns")

        # Load the misfit at each station
        station_misfit_dict = {}
        misfit_group = f["misfit"]
        for station in misfit_group.attrs.keys():
            station_misfit_dict[station] = misfit_group.attrs[station]

        # Load the misfit volume
        easts_grid = f["misfit_volume"]["east_grid"][:]
        norths_grid = f["misfit_volume"]["north_grid"][:]
        depths_grid = f["misfit_volume"]["depth_grid"][:]
        misfit_vol = f["misfit_volume"]["misfit"][:]

        # Pack the location information
        param_dict = {
            "event_type": event_type,
            "event_id": event_id,
            "arrival_type": arrival_type,
            "phase": phase,
            "subarray": subarray,
            "scale_factor": scale_factor,
            "weight": weight
        }

        # Pack the grid information
        grid_dict = {
            "easts_grid": easts_grid,
            "norths_grid": norths_grid,
            "depths_grid": depths_grid,
            "misfit_vol": misfit_vol
        }
    
        return param_dict, location_dict, arrival_time_dict, station_misfit_dict, grid_dict
    


"""
Save the location to an CSV file
"""
def save_location_to_csv_individual(filepath, template_id, arrival_type, phase, scale_factor, weight, east, north, depth, origin_time, misfit):
    subarray = get_template_subarray(template_id)
    
    location_dict = {
        "template_id": template_id,
        "arrival_type": arrival_type,
        "phase": phase,
        "subarray": subarray,
        "scale_factor": scale_factor,
        "weight": weight,
        "east": east,
        "north": north,
        "depth": depth,
        "origin_time": origin_time,
        "misfit": misfit,
    }
    
    location_df = DataFrame(location_dict, index=[0])
    #print(location_df)
    location_df.to_csv(filepath, index=False)
    print(f"Saved the location to {filepath}.")

"""
Plot the misfit distribution of a template event.
The figure consists of two subplots:
- The first subplot shows the misfit distribution at the depth slice of the template event
- The second subplot shows the misfit distribution at the north slice of the template event
"""
def plot_misfit_distribution(misfit_grid, easts_grid, norths_grid, depths_grid, i_east, i_north, i_depth, station_df,
                          misfitmax=0.01,
                          figwidth = 8.0, hspace = 0.07, margin_x = 0.02, margin_y = 0.02,
                          source_size = 300, station_size = 250,
                          cbar_x = 0.05, cbar_y = 0.05, cbar_width = 0.03, cbar_height = 0.3, cbar_tick_spacing = 0.005,
                          title = None):
    
    # Compute the dimensions of the subplots
    min_east = easts_grid.min()
    max_east = easts_grid.max()
    min_north = norths_grid.min()
    max_north = norths_grid.max()
    min_depth = depths_grid.min()
    max_depth = depths_grid.max()

    east_range = max_east - min_east
    north_range = max_north - min_north
    depth_range = max_depth - min_depth

    aspect_map = north_range / east_range
    aspec_profile = depth_range / east_range
    fig_height = figwidth * (1 - 2 * margin_x) * (aspect_map + aspec_profile) / (1 - 2 * margin_y - hspace)

    # Create the figure
    fig = figure(figsize=(figwidth, fig_height))

    # Create the subplot for the profile
    frac_profile_height = (1 - 2 * margin_y - hspace) / (depth_range + north_range) * depth_range
    ax_profile = fig.add_axes([margin_x, margin_y, 1 - 2 * margin_x, frac_profile_height])

    # Create the subplot for the map
    frac_map_height = 1 - 2 * margin_y - hspace - frac_profile_height
    ax_map = fig.add_axes([margin_x, margin_y + frac_profile_height + hspace, 1 - 2 * margin_x, frac_map_height])

    # Plot the misfit distribution at the depth slice
    misfit_map = misfit_grid[i_depth, :, :]
    east_min_rms = easts_grid[i_east]
    north_min_rms = norths_grid[i_north]
    depth_min_rms = depths_grid[i_depth]

    im = ax_map.pcolormesh(easts_grid, norths_grid, misfit_map, vmin=0.0, vmax=misfitmax, shading='auto')
    ax_map.scatter(east_min_rms, north_min_rms, color='salmon', marker='*', linewidths=1.0, edgecolors='black', s=source_size, zorder=10)
    format_east_xlabels(ax_map, 
                        major_tick_spacing=10.0,
                        num_minor_ticks=5)
    format_north_ylabels(ax_map,
                         major_tick_spacing=10.0,
                         num_minor_ticks=5)
    ax_map.set_title(f"Misfit at depth = {depth_min_rms:.0f} m", fontsize=12, fontweight='bold')

    # Plot the stations
    for _, row in station_df.iterrows():
        ax_map.scatter(row["east"], row["north"], color='lightgray', marker='^', linewidths=1.0, edgecolors='black', s=station_size)
        ax_map.annotate(row["name"], (row["east"], row["north"]), xytext=(0, 10), textcoords='offset points', color='black', fontsize=12, ha="center", va="bottom")

    # Plot the misfit distribution along the profile
    misfit_profile = misfit_grid[:, i_north, :]
    ax_profile.pcolormesh(easts_grid, depths_grid, misfit_profile, vmin=0.0, vmax=misfitmax)
    ax_profile.scatter(east_min_rms, depth_min_rms, color='salmon', marker='*', linewidths=1.0, edgecolors='black', s=source_size, zorder=10)
    format_east_xlabels(ax_profile,
                        major_tick_spacing=10.0,
                        num_minor_ticks=5)
    format_depth_ylabels(ax_profile,
                         major_tick_spacing=10.0,
                         num_minor_ticks=5)
    ax_profile.set_title(f"Misfit at north = {north_min_rms:.0f} m", fontsize=12, fontweight='bold')

    ax_profile.set_xlim(min_east, max_east)
    ax_profile.set_ylim(min_depth, max_depth)
    ax_profile.invert_yaxis()

    # Plot the colorbar
    cbar_ax = ax_profile.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar = colorbar(im, cax=cbar_ax, orientation='vertical', label="Misfit (s)")
    cbar.set_ticks(arange(0, misfitmax + cbar_tick_spacing, cbar_tick_spacing))

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    return fig, ax_map, ax_profile, cbar

"""
Process the arrival information stored in a data frame
"""
def process_arrival_info(arrival_df, arrival_type):
    arrival_df = arrival_df.copy()
    
    # Format the arrival times and uncertainty
    if arrival_type == "kurtosis_stack" or arrival_type == "kurtosis" or arrival_type == "sta_lta_stack" or arrival_type == "sta_lta":
        arrival_df["arrival_time"] = arrival_df["arrival_time"].apply(lambda x: x.timestamp()) # Convert to seconds since the Unix epoch
    elif arrival_type == "manual_stack":
            arrival_df["starttime"] = arrival_df["starttime"].apply(lambda x: x.timestamp())
            arrival_df["endtime"] = arrival_df["endtime"].apply(lambda x: x.timestamp())
            arrival_df["arrival_time"] = arrival_df["starttime"] + arrival_df["duration"] / 2
            arrival_df["uncertainty"] = arrival_df["duration"] / 2

    return arrival_df

"""
Get the misfit and origin time grids
"""
def get_misfit_and_origin_time_grids(arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict, weight = False):
    # Loop through the grid points
    misfit_vol = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    origin_times_grid = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    print(f"Computing RMS and origin time grids...")
    for i_depth, _ in tqdm(enumerate(depths_grid), total=len(depths_grid), desc="Depth"):
        for i_north, _ in enumerate(norths_grid):
            for i_east, _ in enumerate(easts_grid):
                misfit, origin_time = get_misfit_and_origin_time_for_grid_point(arrival_df, i_east, i_north, i_depth, travel_time_dict, weight)
                misfit_vol[i_depth, i_north, i_east] = misfit
                origin_times_grid[i_depth, i_north, i_east] = origin_time

    # Replace the nan values with inf
    misfit_vol[isnan(misfit_vol)] = inf

    return misfit_vol, origin_times_grid

"""
Get the misfit and origin time for one grid point
"""
def get_misfit_and_origin_time_for_grid_point(arrival_df, i_east, i_north, i_depth, travel_time_dict):
    # Loop through the arrival times
    origin_times = zeros(len(arrival_df))
    travel_times = zeros(len(arrival_df))
    arrival_times = zeros(len(arrival_df))
    for i, row in arrival_df.iterrows():
        station = row["station"]
        travel_time = travel_time_dict[station][i_depth, i_north, i_east]
        arrival_time = row["arrival_time"]

        origin_time = arrival_time - travel_time
        origin_times[i] = origin_time
        travel_times[i] = travel_time
        arrival_times[i] = arrival_time

    origin_time = mean(origin_times)
    misfit = sum(abs(arrival_times - travel_times - origin_time) / weights) / sum(1 / weights)

    return misfit, origin_time


    
# # Compute the body-wave travel time between a certain station and every grid point in 3D using a homogeneous velocity model
# def get_station_traveltimes_3dgrid_homo(steast, stnorth, vel, numeast = 61, numnorth = 61, numdep = 26, eastmin=-120, eastmax=120, northmin=-120, northmax=120, depmin=0, depmax=50):

#     ## Build the meshgrid
#     traveltime = zeros((numdep, numnorth, numeast))
#     depgrid = linspace(depmin, depmax, num=numdep)
#     nogrid = linspace(northmin, northmax, num=numnorth)
#     eagrid = linspace(eastmin, eastmax, num=numeast)

#     for depind in range(numdep):
#         grddep = depgrid[depind]

#         for northind in range(numnorth):
#             grdnorth = nogrid[northind]

#             for eastind in range(numeast):
#                 grdeast = eagrid[eastind]

#                 dist = norm(array([steast, stnorth, 0])-array([grdeast, grdnorth, grddep]))
#                 traveltime[depind, northind, eastind] = dist/vel

#     return traveltime, depgrid, nogrid, eagrid

# # Compute the surface-wave travel time between a certain station and every grid point in 2D using a homogeneous velocity model
# def get_station_traveltimes_2dgrid_homo(steast, stnorth, vel, numeast = 61, numnorth = 61, eastmin=-120, eastmax=120, northmin=-120, northmax=120):
#     from numpy import sqrt, zeros, array, linspace
#     from numpy.linalg import norm

#     ## Build the meshgrid
#     traveltime = zeros((numnorth, numeast))
#     numgrid = numeast*numnorth
#     nogrid = linspace(northmin, northmax, num=numnorth)
#     eagrid = linspace(eastmin, eastmax, num=numeast)

#     for northind in range(numnorth):
#         grdnorth = nogrid[northind]

#         for eastind in range(numeast):
#             grdeast = eagrid[eastind]

#             dist = norm(array([steast, stnorth])-array([grdeast, grdnorth]))
#             traveltime[northind, eastind] = dist/vel

#     return traveltime, nogrid, eagrid


# # Find the event location and origin time through 3D grid search
# def locate_event_3dgrid(pickdf, ttdict, depgrid, nogrid, eagrid):


#     begin = time()

#     numpk = pickdf.shape[0]
#     numdep = len(depgrid)
#     numnor = len(nogrid)
#     numeas = len(eagrid)

#     rmsvol = zeros((numdep, numnor, numeas))
#     orivol = zeros((numdep, numnor, numeas))

#     for depind in range(numdep):
#         for norind in range(numnor):
#             for easind in range(numeas):
#                 ### Compute the origin time       
#                 otimes_sta = zeros(numpk)
#                 ttimes = zeros(numpk)
#                 atimes = zeros(numpk)
#                 for staind, row in pickdf.iterrows():
#                     stname = row['station']
#                     atime = UTCDateTime(row['time'])

#                     ttime = ttdict[stname][depind, norind, easind]
#                     otime = atime-ttime
#                     otimes_sta[staind] = otime.timestamp
#                     ttimes[staind] = ttime
#                     atimes[staind] = atime.timestamp

#                 otime = mean(otimes_sta)
#                 orivol[depind, norind, easind] = otime
            
#                 ### Compute the RMS
#                 rms = 0
#                 for staind, ttime in enumerate(ttimes):
#                     ttime = ttimes[staind]
#                     atime = atimes[staind]
#                     rms = rms+(atime-otime-ttime)**2

#                 rms = sqrt(rms/numpk)
#                 rmsvol[depind, norind, easind] = rms


#     ## Find the grid with the smallest RMS
#     ind = argmin(rmsvol)
#     evdpind, evnoind, eveaind = unravel_index(ind, rmsvol.shape)
#     evori = UTCDateTime(orivol[evdpind, evnoind, eveaind])

#     ## Compute the predicted arrival times
#     atimedict = {}
#     for staind, row in pickdf.iterrows():
#         stname = row['station']
#         atime = UTCDateTime(row['time'])

#         ttime = ttdict[stname][evdpind, evnoind, eveaind]
#         otime = evori
#         atime = otime+ttime
#         atimedict[stname] = atime

#     end = time()

#     end = time()
#     print(f"Time elapsed: {end-begin:1f} s.")

#     return evdpind, evnoind, eveaind, evori, rmsvol, atimedict

# # Plot the RMS distributions
# def plot_rms(rmsvol, evdpind, evnoind, eveaind, depgrid, northgrid, eastgrid, stadf, pickdf, evname, rmsmax=0.02):
#     import matplotlib.pyplot as plt
#     import numpy as np

#     eastmin = eastgrid[0]
#     eastmax = eastgrid[-1]
#     northmin = northgrid[0]
#     northmax = northgrid[-1]
#     depmax = depgrid[-1]

#     evea = eastgrid[eveaind]
#     evno = northgrid[evnoind]
#     evdp = depgrid[evdpind]

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10), gridspec_kw={'width_ratios': [1]})
#     ax1.imshow(rmsvol[evdpind, :, :], extent=[eastmin, eastmax, northmin, northmax], origin='lower', vmin=0.0, vmax=rmsmax, aspect='equal')
    
#     for _, row in stadf.iterrows():
#         stname = row['name']
#         east = row['east']
#         north = row['north']

#         if eastmin < east < eastmax and northmin < north < northmax:
#             if stname in pickdf['station'].values:
#                 ax1.scatter(east, north, color='white', marker='^')
#                 ax1.text(east, north+0.5, stname, color='white')
#             else:
#                 ax1.scatter(east, north, color='gray', marker='^')

#     ax1.scatter(evea, evno, color='red', marker='*')
#     ax1.set_ylabel("North (m)")
#     ax1.set_title(f"{evname}, RMS at depth {evdp:.0f} m", fontweight='bold')

#     imag2 = ax2.imshow(rmsvol[:, evnoind, :], extent=[eastmin, eastmax, depmax, 0], origin='upper', vmin=0.0, vmax=rmsmax, aspect='equal')
#     ax2.scatter(evea, evdp, color='red', marker='*')
#     ax2.set_title(f"RMS at north {evno:.0f} m", fontweight='bold')

#     ax2.set_xlabel("East (m)")
#     ax2.set_ylabel("Depth (m)")

#     ax3 = fig.add_axes([0.15, 0.3, 0.01, 0.08])
#     fig.colorbar(imag2, cax=ax3, orientation='vertical', label='RMS (s)')

#     return fig, ax1, ax2, ax3
#     #return fig, ax1, ax2

# Get the apparent velocity for a station triad using the phase differences of two station pairs
# The phase differences are in radians
def get_triad_app_vel(time_diff_12, time_diff_23, dist_mat_inv):
    """
    Get the apparent velocity for a station triad using the phase differences of two station pairs

    Parameters
    ----------
    time_diff_12 : array_like
        The time difference between the first and second station
    time_diff_23 : array_like
        The time difference between the second and third station
    dist_mat_inv : array_like
        The inverse of the distance matrix

    Returns
    -------
    vel_app : float
        The apparent velocity
    back_azi : float
        The back azimuth in radians
    vel_app_east : float
        The apparent velocity component in the east direction
    vel_app_north : float
        The apparent velocity component in the north direction
    """

    # Compute the slowness vector
    slow_vec = dist_mat_inv @ array([time_diff_12, time_diff_23])
    slow_vec = slow_vec.flatten()

    # Compute the apparent velocity
    vel_app = 1 / norm(slow_vec)

    # Compute the back azimuth
    back_azi = arctan2(slow_vec[0], slow_vec[1])

    # Compute the apparent velocity components
    vel_app_east = vel_app * sin(back_azi)
    vel_app_north = vel_app * cos(back_azi)

    return vel_app, back_azi, vel_app_east, vel_app_north

# Get the inverse of the distance matrix
def get_dist_mat_inv(east1, north1, east2, north2, east3, north3):
    """
    Get the inverse of the distance matrix

    Parameters
    ----------
    east1 : float
        The east coordinate of the first station
    north1 : float
        The north coordinate of the first station
    east2 : float
        The east coordinate of the second station
    north2 : float
        The north coordinate of the second station
    east3 : float
        The east coordinate of the third station
    north3 : float
        The north coordinate of the third station
    """

    dist_vec_12 = array([east2 - east1, north2 - north1])
    dist_vec_23 = array([east3 - east2, north3 - north2])
    dist_mat = vstack([dist_vec_12, dist_vec_23])

    dist_mat_inv = inv(dist_mat)

    return dist_mat_inv
