# Compute sunrise and sunset times for the hydrophone deployment
# Imports
from os.path import join
from astral import LocationInfo
from astral.sun import sun
from pandas import Timestamp, DataFrame

from utils_basic import CENTER_LONGITUDE as longitude, CENTER_LATITUDE as latitude, TIME_DIR as outdir
from utils_basic import get_hydrophone_days

# Get the days of the geophone deployment
days = get_hydrophone_days(timestamp = True)

# Get the location information
location = LocationInfo(longitude = longitude, latitude = latitude)

# Compute the sunrise and sunset times
sunrise_times = []
sunset_times = []

for day in days:
    s = sun(location.observer, date = day)
    sunrise_time = Timestamp(s["sunrise"])
    sunset_time = Timestamp(s["sunset"])
    sunrise_times.append(sunrise_time)
    sunset_times.append(sunset_time)

# Construct the output DataFrame
out_df = DataFrame({"day": days, "sunrise": sunrise_times, "sunset": sunset_times})
out_df.set_index("day", inplace = True)

# Save the output
outpath = join(outdir, "sunrise_sunset_times.csv")
out_df.to_csv(outpath, index = True)
print(f"Saved sunrise and sunset times to {outpath}")
