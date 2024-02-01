# Functions and classes for analyzing seismic data using the STA/LTA method


## Import libraries
from numpy import sqrt, mean, square, amin, argmin
from obspy.signal.trigger import coincidence_trigger
from pandas import DataFrame, Timedelta, Grouper, concat, read_csv
from matplotlib.pyplot import subplots
from matplotlib.dates import DayLocator, DateFormatter

from utils_basic import INNER_STATIONS, DAYS_PATH, NIGHTS_PATH, STARTTIME, ENDTIME

## Class for storing the information of an event claimed by associating the detections
class AssociatedEvent:
    def __init__(self, name, stations, trigger_times, detrigger_times, snrs):
        self.name = name
        self.stations = stations
        self.trigger_times = trigger_times
        self.detrigger_times = detrigger_times
        self.snrs = snrs

        self.num_of_stations = len(stations)
        self.first_trigger_time = amin(trigger_times)
        self.first_trigger_station = stations[argmin(trigger_times)]

        if snrs is not None:
            self.average_snr = mean(snrs)
        else:
            self.average_snr = None

    def __str__(self):
        return f"Event {self.name} is detected by {self.num_of_stations} stations. The first trigger time is {self.first_trigger_time}. The average SNR is {self.average_snr}."
    
    def __repr__(self):
        return self.__str__()
    
    def append_to_file(self, fp):
        fp.write("##\n")
        fp.write(f"{self.name}\n")
        fp.write("\n")
        fp.write(f"first_trigger_time first_trigger_station num_of_stations average_snr\n")

        timestr = self.first_trigger_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

        if self.average_snr is not None:
            fp.write(f"{timestr} {self.first_trigger_station} {self.num_of_stations:d} None\n")
        else:
            fp.write(f"{timestr} {self.first_trigger_station} {self.num_of_stations:d} None\n")
        fp.write("\n")

        fp.write("station trigger_time detrigger_time signal_noise_ratio\n")
        
        for i in range(self.num_of_stations):
            station = self.stations[i]
            trigger_time = self.trigger_times[i]
            detrigger_time = self.detrigger_times[i]
            timestr1 = trigger_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
            timestr2 = detrigger_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

            if self.snrs is not None:
                snr = self.snrs[i]
                
                fp.write(f"{station} {timestr1} {timestr2} {snr:.2f}\n")
            else:
                fp.write(f"{station} {timestr2} {timestr2} None\n")

        fp.write("\n")

## Class for storing a list of AssociatedEvent objects  
class Events:
    def __init__(self, min_num_det, max_delta, min_snr=None):
        self.min_num_det = min_num_det
        self.max_delta = max_delta
        self.min_snr = min_snr
        self.events = []

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __setitem__(self, index, event):
        if not isinstance(event, AssociatedEvent):
            raise ValueError("Only objects of AssociatedEvent type can be added to the list.")
        self.events[index] = event

    def __delitem__(self, index):
        del self.events[index]

    def append(self, event):
        if not isinstance(event, AssociatedEvent):
            raise ValueError("Only objects of AssociatedEvent type can be added to the list.")
        self.events.append(event)

    def remove(self, event):
        self.events.remove(event)

    def clear(self):
        self.events.clear()

    def __str__(self):
        return f"Number of events: {len(self.events)}"

    def __repr__(self):
        return self.__str__()
    
    def get_first_trigger_times(self):
        trigger_times = []
        for event in self.events:
            trigger_times.append(event.first_trigger_time)
        return trigger_times
    
    def write_to_file(self, path):
        with open(path, "w") as fp:
            fp.write("#\n")
            fp.write("Parameters\n")
            fp.write("\n")
            fp.write("min_num_of_detections max_delta min_snr\n")

            if self.min_snr is None:
                fp.write(f"{self.min_num_det:d} {self.max_delta:.3f} None\n")
            else:
                fp.write(f"{self.min_num_det:d} {self.max_delta:.3f} {self.min_snr:.1f}\n")

            fp.write("\n")

            for event in self.events:
                event.append_to_file(fp)
        
        print(f"Events are written to {path}.")

### Run the STA/LTA method on the 3C waveforms of a given station at a given time window
def run_sta_lta(stream, sta, lta, thr_on, thr_off, thr_coincidence_sum=2, trigger_type='classicstalta'):

    #### Determine if the number of channels is correct
    if len(stream) != 3:
        raise ValueError("The input stream must have 3 channels!")

    #### Run the STA/LTA method
    triggers = coincidence_trigger(trigger_type=trigger_type, thr_on=thr_on, thr_off=thr_off, stream=stream, thr_coincidence_sum=thr_coincidence_sum, sta=sta, lta=lta)

    #### Eliminate repeating detections (Why do they exist? Bug reported on Github, 2023-10-17.)
    triggers, numrep = remove_repeating_triggers(triggers)

    numdet = len(triggers)
    print("Finished.")
    print(f"Number of detections: {numdet}. Number of repeating detections removed: {numrep}.")

    return triggers


### Eliminate repeating STA/LTA detections (Why do they exist? Bug reported on Github, 2023-10-17.)
def remove_repeating_triggers(triggers):

    numrep = 0
    i = 0
    print("Eliminating repeating events...")
    while i < len(triggers)-1:
        trigtime1 = triggers[i]['time']
        dur = triggers[i]['duration'] 
        trigtime2 = triggers[i+1]['time']

        if trigtime2-trigtime1 < dur:
            numrep = numrep+1
            triggers.pop(i+1)
        else:
            i = i+1
    
    return triggers, numrep

### Get the SNR of a given detection
def get_snr(trigger, stream, snrwin):

    starttime_noi = trigger['time'] - snrwin
    endtime_noi = trigger['time']

    starttime_sig = trigger['time']
    endtime_sig = trigger['time'] + snrwin
        
    trace_z = stream.select(channel='GHZ')[0]
    trace_1 = stream.select(channel='GH1')[0]
    trace_2 = stream.select(channel='GH2')[0]

    sigtrace_z = trace_z.copy()
    sigtrace_z.trim(starttime_sig, endtime_sig)
    signal_z = sigtrace_z.data

    noitrace_z = trace_z.copy()
    noitrace_z.trim(starttime_noi, endtime_noi)
    noise_z = noitrace_z.data

    snr_z = sqrt(mean(square(signal_z)))/sqrt(mean(square(noise_z)))

    sigtrace_1 = trace_1.copy()
    sigtrace_1.trim(starttime_sig, endtime_sig)
    signal_1 = sigtrace_1.data

    noitrace_1 = trace_1.copy()
    noitrace_1.trim(starttime_noi, endtime_noi)
    noise_1 = noitrace_1.data

    snr_1 = sqrt(mean(square(signal_1)))/sqrt(mean(square(noise_1)))

    sigtrace_2 = trace_2.copy()
    sigtrace_2.trim(starttime_sig, endtime_sig)
    signal_2 = sigtrace_2.data

    noitrace_2 = trace_2.copy()
    noitrace_2.trim(starttime_noi, endtime_noi)
    noise_2 = noitrace_2.data

    snr_2 = sqrt(mean(square(signal_2)))/sqrt(mean(square(noise_2)))     

    snr = mean([snr_z, snr_1, snr_2])

    return snr

## Merge the detections from different stations into a single dataframe
## separate: if True, each subarray will have its own dataframe
def merge_station_detections(stadet_dict, separate=True):
    
    if separate:
        detdfs_a = []
        detdfs_b = []
        for key in stadet_dict.keys():
            if key.startswith("A"):
                detdf = stadet_dict[key].copy()
                detdf["station"] = key
                detdfs_a.append(detdf)
            elif key.startswith("B"):
                detdf = stadet_dict[key].copy()
                detdf["station"] = key
                detdfs_b.append(detdf)

        mergeddf_a = concat(detdfs_a, ignore_index=True)
        mergeddf_a.sort_values(by="trigger_time", inplace=True)
        mergeddf_a.reset_index(drop=True, inplace=True)

        mergeddf_b = concat(detdfs_b, ignore_index=True)
        mergeddf_b.sort_values(by="trigger_time", inplace=True)
        mergeddf_b.reset_index(drop=True, inplace=True)

        print(f"Total number of detections to associate: {len(mergeddf_a)} (A) and {len(mergeddf_b)} (B).")

        return mergeddf_a, mergeddf_b
    else:
        detdfs = []
        for key in stadet_dict.keys():
            detdf = stadet_dict[key].copy()
            detdf["station"] = key
            detdfs.append(detdf)

        mergeddf = concat(detdfs, ignore_index=True)
        mergeddf.sort_values(by="trigger_time", inplace=True)
        mergeddf.reset_index(drop=True, inplace=True)

        print(f"Total number of detections to associate: {len(mergeddf)}.")

        return mergeddf

## Find the detected events by associating the detections
def associate_detections(detdf, numdet_min=3, delta_max=0.1, use_snr=False, min_snr=1.0):

    ### Associate the detections
    numdet = len(detdf)
    evdfs = []
    i = 0
    while i < numdet:
        if use_snr:
            snr = detdf["signal_noise_ratio"][i]

            if snr < min_snr:
                i += 1
                continue
        
        station0 = detdf["station"][i]
        trigger_time0 = detdf["trigger_time"][i]

        stations = [station0]
        detdfs_tmp = [detdf.iloc[i]]
        j = i + 1

        while j < numdet:
            if use_snr:
                snr = detdf["signal_noise_ratio"][j]

                if snr < min_snr:
                    j += 1
                    continue

            trigger_time = detdf["trigger_time"][j]
            station = detdf["station"][j]
            delta = trigger_time - trigger_time0

            if delta.total_seconds() > delta_max:
                break
            
            if station not in stations:
                stations.append(station)
                detdfs_tmp.append(detdf.iloc[j])
            
            j += 1

        if len(stations) >= numdet_min:
            evdf = DataFrame(detdfs_tmp)
            evdf.reset_index(drop=True, inplace=True)
            evdfs.append(evdf)
            i = j
        else:
            i += 1

    numev = len(evdfs)
    print(f"There are in total {numev} associated events.")

    ### Store the information in an Events object
    if use_snr:
        events = Events(numdet_min, delta_max, min_snr)
    else:
        events = Events(numdet_min, delta_max)

    for i, evdf in enumerate(evdfs):
        evname = "Event" + str(i+1).zfill(len(str(numev)))
        stations = evdf["station"].tolist()

        if use_snr:
            snrs = evdf["signal_noise_ratio"].array
        else:
            snrs = None

        trigger_times = evdf["trigger_time"].array
        detrigger_times = evdf["detrigger_time"].array
        event = AssociatedEvent(evname, stations, trigger_times, detrigger_times, snrs)
        events.append(event)

    return events

## Bin the associated events by hour
def bin_events_by_hour(events):
    timedf = DataFrame(events.get_first_trigger_times(), columns=["trigger_time"])
    timedf.set_index("trigger_time", inplace=True)

    countdf = DataFrame(timedf.groupby(Grouper(freq="H")).size())
    countdf.rename(columns={0: "count"}, inplace=True)
    countdf.index.name = "hour"

    return countdf

## Plot the number of detections for each station in Subarray A and B
def plot_station_hourly_detections(detnumdf, individual_color=True, days_and_nights=False):
    fig, axes = subplots(2, 1, sharex=True, figsize=(20, 12))

    ### Plot blocks for days and nights
    if days_and_nights:
        daysdf = read_csv(DAYS_PATH, parse_dates=["starttime", "endtime"])
        nightsdf = read_csv(NIGHTS_PATH, parse_dates=["starttime", "endtime"])

        for i in range(len(daysdf)):
            sunrise = daysdf["starttime"].iloc[i]
            sunset = daysdf["endtime"].iloc[i]

            axes[0].axvspan(sunrise, sunset, color="lightyellow", alpha=0.5)
            axes[1].axvspan(sunrise, sunset, color="lightyellow", alpha=0.5)

        for i in range(len(nightsdf)):
            sunset = nightsdf["starttime"].iloc[i]
            sunrise = nightsdf["endtime"].iloc[i]

            axes[0].axvspan(sunset, sunrise, color="lightblue", alpha=0.5)
            axes[1].axvspan(sunset, sunrise, color="lightblue", alpha=0.5)

    timeax = detnumdf["hour"].values

    stadf = detnumdf.iloc[:, 1:]
    for station, data in stadf.items():
        if individual_color:
            if station.startswith("A"):
                if station in INNER_STATIONS:
                    axes[0].plot(timeax, data, color="dodgerblue", label=station, linewidth=0.5)
                else:
                    axes[0].plot(timeax, data, color="dodgerblue", label=station, linestyle=":", linewidth=0.5) 
            else:
                if station in INNER_STATIONS:
                    axes[1].plot(timeax, data, color="darkorange", label=station, linewidth=0.5)
                else:
                    axes[1].plot(timeax, data, color="darkorange", label=station, linestyle=":", linewidth=0.5)
        else:
            if station in INNER_STATIONS:
                axes[0].plot(timeax, data, color="lightgray", label=station, linewidth=0.5)
                axes[1].plot(timeax, data, color="lightgray", label=station, linewidth=0.5)
            else:
                axes[0].plot(timeax, data, color="lightgray", label=station, linestyle=":", linewidth=0.5)
                axes[1].plot(timeax, data, color="lightgray", label=station, linestyle=":", linewidth=0.5)

    axes[0].set_ylabel("Detections per hour", fontsize=15)
    axes[1].set_ylabel("Detections per hour", fontsize=15)

    axes[0].set_title("Array A", fontsize=18, fontweight="bold")
    axes[1].set_title("Array B", fontsize=18, fontweight="bold")

    ## Set x ticks and labels
    axes[1].tick_params(axis='x', which='major', length=5)
    axes[1].xaxis.set_major_locator(DayLocator(interval=1))
    axes[1].xaxis.set_major_formatter(DateFormatter("%m-%d"))

    for label in axes[1].get_xticklabels():
        label.set_va('top')  # Vertical alignment for y-axis label
        label.set_ha('right')
        label.set_rotation(15)
        label.set_fontsize(12)

    for label in axes[0].get_yticklabels():
        label.set_fontsize(12)

    for label in axes[1].get_yticklabels():
        label.set_fontsize(12)

    axes[1].set_xlabel("UTC Time", fontsize=15)
    axes[1].set_xlim(STARTTIME, ENDTIME)

    return fig, axes


