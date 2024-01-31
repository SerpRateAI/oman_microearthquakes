# Functions and classes for analyzing seismic data using the STA/LTA method

## Import libraries
from numpy import sqrt, mean, square
from obspy.signal.trigger import coincidence_trigger

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