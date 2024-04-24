# Documentation for the codes developed for analyzing geophone and hydrophone data collected at the Oman Multi-borehole Observatory
*Tianze Liu* <tianzeliu@gmail.com>
*John Mark Aiken* <john@xal.no>


## Spectrograms
### Overview
Spectrograms are the main data products of the project and are computed using short-time Fourier transform (STFT). To speed up the computation, we use the STFT function of PyTorch to utilize its CPU- and GPU-based parallel computing capacities.

### Computing spectrograms
To be done...

### Reading and writing spectrograms
#### How spectrograms are stored
Spectrograms are stored in HDF5 format (file extension `.h5`) in the directory specified through the constant `SPECTROGRAM_DIR` defined in `utils_basic.py`. Daily and hourly spectrograms are typically stored in separate folders under the directory. The geophone and hydrophone spectrograms typically have the following file names:

`daily_geo_spectrograms_20200113_A01_window60s_overlap0.0.h5`

`daily_hydro_spectrograms_20200113_A00_window60s_overlap0.0.h5`

`daily_geo_spectrograms_20200113_A01_window60s_overlap0.0_downsample60.h5`

`daily_hydro_spectrograms_20200113_A00_window60s_overlap0.0_downsample60.h5`

`hourly_geo_spectrograms_20200113070000_A01_window1s_overlap0.0.h5`

`hourly_hydro_spectrograms_20200113070000_A00_window1s_overlap0.0.h5`

The daily and hourly spectrograms typically have time-window lengths of 1 minuite and 1 second, respectively. Each daily spectrogram has two copies, one with a full frequency resolution (~0.017 Hz) and the other downsampled to having the same frequency resolution as the hourly ones (~1 Hz). The full-resolution ones are for further analyses, and the downsampled ones are for plotting.

Each geophone spectrogram file contains all three components of the station, whereas each hydrohpone file contains all locations (each phone is represented by a location) of the station. The component and location information is stored in the headers of the HDF files. The First two phones of A00 are considered broken and thus are not stored.

#### How to read a spectrogram file
Geophone and hydrophone spectrograms are read into Python as `utils_spec.StreamSTFTPSD` objects with `utils_spec.read_geo_spectrograms()` and `utils_spec.read_hydro_spectrograms()`, respectively. A `StreamSTFTPSD` object is a list-like object containing multiple `utils_spec.TraceSTFTPSD` objects. The attributes of a `TraceSTFTPSD` object are listed below:

`station`: Station name.

`location`: Location code (`None` for geophone data)

`component`: Component code (`"Z"`, `"1"`, and `"2"` for geohpones and `"H"` for hydrophones)

`time_label`: String in the format `"%Y%m%d%H%M%S"` representing the intended startting time of the spectrograms used for organizing spectrograms, although the true starting time of the spectrogram (first element of `TraceSTFTPSD.times`) is always later.

`times`: `pandas.DatetimeIndex` object containing the time for each column of the data matrix.

`freqs`: `numpy.array` object containing the frequency for each row of the data matrix.

`overlap`: Fraction of overlap between adjacent time windows.

`data`: `numpy.array` object containing the power-spectral density for each frequency and time.

`db`: Whether the `data` is in dB (`True`) or not (`False`).



### Plotting spectrograms
