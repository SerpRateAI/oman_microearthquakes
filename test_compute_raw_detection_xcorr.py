"""
Test the efficiency of computing the correlation between one of the raw STA/LTA detection with all others
"""

# Inport libararies
from os.path import join
from time import time
from numpy import correlate
from numpy.linalg import norm

from utils_basic import DETECTION_DIR as dirpath
from utils_sta_lta import Detections

# Input parameters
station = 'A04'
component = 'Z'
thr_on = 17.0

# Read the detections
filename = f'raw_sta_lta_detections_{station}_{component.lower()}_on{thr_on:.1f}.h5'
inpath = join(dirpath, filename)

print(f'Reading the detections from {inpath}...')
clock1 = time()
detections = Detections.from_hdf(inpath)
num_det = len(detections)
clock2 = time()
elapsed = clock2 - clock1
print(f'Time taken: {elapsed} s.')
print(f'{num_det} detections are read.')


# Compute the correlation
## Extract the master waveform
print(f'Extracting the master waveform')
detection_master = detections[0]
waveform_master = detection_master.waveform

## Loop over all other detections
print(f'Looping over the rest of the  the detections from {inpath}...')
clock1 = time()
for i, detection in enumerate(detections[1:]):
    waveform = detection.waveform 

    ### Compute the normalized cross-correlation
    xcorr = correlate(waveform_master, waveform, mode="full").max()
    denom = norm(waveform) * norm(waveform_master)
    xcorr /= denom

    ### Print the progress
    if (i + 1) % 1000 == 0:
        print(f'{i+1} cross-correlations have bee computed.')
clock2 = time()
elapsed = clock2 - clock1
print(f'Time taken: {elapsed} s.')

