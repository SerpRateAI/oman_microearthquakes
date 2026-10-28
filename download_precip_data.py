"""
Download hourly ERA5 total precipitation for the study site from the CDS API.
"""

from argparse import ArgumentParser
from os import makedirs, remove
from os.path import exists, join
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, is_zipfile

from cdsapi import Client
from pandas import Timestamp, concat, date_range, to_datetime
from pandas.tseries.offsets import MonthEnd
from xarray import open_dataset

from utils_basic import CENTER_LATITUDE, CENTER_LONGITUDE, ROOTDIR_OTHER

CDS_DATASET = "reanalysis-era5-single-levels-timeseries"
DEFAULT_STARTTIME = "2019-05-01"
DEFAULT_ENDTIME = "2020-02-29"
DEFAULT_OUTFILE = "era5_hourly_precipitation.csv"
RAW_NETCDF_NAME = "era5_hourly_precipitation_raw.nc"


def parse_args():
    parser = ArgumentParser(
        description="Download hourly ERA5 precipitation for the study site.",
    )
    parser.add_argument("--latitude", type=float, default=CENTER_LATITUDE)
    parser.add_argument("--longitude", type=float, default=CENTER_LONGITUDE)
    parser.add_argument("--starttime", type=str, default=DEFAULT_STARTTIME)
    parser.add_argument("--endtime", type=str, default=DEFAULT_ENDTIME)
    parser.add_argument("--outdir", type=str, default=ROOTDIR_OTHER)
    parser.add_argument("--outfile", type=str, default=DEFAULT_OUTFILE)
    parser.add_argument(
        "--keep-netcdf",
        action="store_true",
        help="Retain the raw NetCDF file after writing CSV.",
    )
    return parser.parse_args()


def month_ranges(starttime, endtime):
    start = Timestamp(starttime)
    end = Timestamp(endtime)
    month_starts = date_range(start=start, end=end, freq="MS")

    ranges = []
    for month_start in month_starts:
        month_end = month_start + MonthEnd(0)
        chunk_start = max(month_start, start)
        chunk_end = min(month_end, end)
        ranges.append(
            (
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )
    return ranges


def build_request(latitude, longitude, start_date, end_date):
    return {
        "variable": ["total_precipitation"],
        "location": {"latitude": latitude, "longitude": longitude},
        "date": [f"{start_date}/{end_date}"],
        "data_format": "netcdf",
    }


class CdsLicenseError(RuntimeError):
    pass


def retrieve_netcdf(client, latitude, longitude, start_date, end_date, target):
    request = build_request(latitude, longitude, start_date, end_date)
    print(
        f"Requesting ERA5 precipitation for ({latitude}, {longitude}) "
        f"from {start_date} to {end_date}..."
    )
    try:
        client.retrieve(CDS_DATASET, request, target)
    except Exception as exc:
        message = str(exc).lower()
        if "licen" in message or "403" in message:
            raise CdsLicenseError(
                "CDS dataset license not accepted. Visit "
                "https://cds.climate.copernicus.eu/datasets/"
                "reanalysis-era5-single-levels-timeseries?tab=download"
                "#manage-licences to accept the required license, then rerun."
            ) from exc
        raise RuntimeError(
            "CDS request failed. Ensure the CDS API token is configured "
            "and network access is available."
        ) from exc


def resolve_netcdf_path(download_path):
    if not is_zipfile(download_path):
        return download_path, []

    with ZipFile(download_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".nc"))
        with NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
            tmp_file.write(archive.read(member))
            return tmp_file.name, [tmp_file.name]


def netcdf_to_dataframe(netcdf_path):
    nc_path, cleanup_paths = resolve_netcdf_path(netcdf_path)
    dataset = None
    try:
        dataset = open_dataset(nc_path, engine="netcdf4")
        if "tp" in dataset:
            precip = dataset["tp"]
        elif "total_precipitation" in dataset:
            precip = dataset["total_precipitation"]
        else:
            raise KeyError(
                "Could not find precipitation variable in NetCDF "
                f"(available: {list(dataset.data_vars)})"
            )

        if "valid_time" not in precip.coords and "time" not in precip.coords:
            raise KeyError(
                "Could not find time coordinate in NetCDF "
                f"(available: {list(precip.coords)})"
            )

        grid_lat = float(dataset["latitude"].values)
        grid_lon = float(dataset["longitude"].values)

        precip_mm = precip.squeeze().to_pandas() * 1000.0
        precip_df = precip_mm.to_frame(name="precipitation_mm")
        precip_df.index = to_datetime(precip_df.index, utc=True)
        precip_df.index.name = "time"
        precip_df = precip_df.sort_index()
        precip_df = precip_df[~precip_df.index.duplicated(keep="first")]
        return precip_df, grid_lat, grid_lon
    finally:
        if dataset is not None:
            dataset.close()
        for path in cleanup_paths:
            if exists(path):
                remove(path)


def download_precipitation(
    client,
    latitude,
    longitude,
    starttime,
    endtime,
    outdir,
):
    makedirs(outdir, exist_ok=True)
    raw_netcdf_path = join(outdir, RAW_NETCDF_NAME)

    try:
        retrieve_netcdf(
            client,
            latitude,
            longitude,
            starttime,
            endtime,
            raw_netcdf_path,
        )
        precip_df, grid_lat, grid_lon = netcdf_to_dataframe(raw_netcdf_path)
        return precip_df, grid_lat, grid_lon, raw_netcdf_path
    except CdsLicenseError:
        if exists(raw_netcdf_path):
            remove(raw_netcdf_path)
        raise
    except RuntimeError:
        if exists(raw_netcdf_path):
            remove(raw_netcdf_path)
        print("Full-range request failed; retrying with monthly chunks...")
        chunk_frames = []
        grid_lat = None
        grid_lon = None

        for chunk_start, chunk_end in month_ranges(starttime, endtime):
            with NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                chunk_path = tmp_file.name

            try:
                retrieve_netcdf(
                    client,
                    latitude,
                    longitude,
                    chunk_start,
                    chunk_end,
                    chunk_path,
                )
                chunk_df, chunk_lat, chunk_lon = netcdf_to_dataframe(chunk_path)
                chunk_frames.append(chunk_df)
                if grid_lat is None:
                    grid_lat = chunk_lat
                    grid_lon = chunk_lon
            except CdsLicenseError:
                raise
            finally:
                if exists(chunk_path):
                    remove(chunk_path)

        if not chunk_frames:
            raise RuntimeError("No precipitation data were downloaded.")

        precip_df = concat(chunk_frames)
        precip_df = precip_df[~precip_df.index.duplicated(keep="first")]
        precip_df = precip_df.sort_index()
        return precip_df, grid_lat, grid_lon, None


def main():
    args = parse_args()
    client = Client()

    precip_df, grid_lat, grid_lon, raw_netcdf_path = download_precipitation(
        client=client,
        latitude=args.latitude,
        longitude=args.longitude,
        starttime=args.starttime,
        endtime=args.endtime,
        outdir=args.outdir,
    )

    outpath = join(args.outdir, args.outfile)
    precip_df.reset_index().to_csv(outpath, index=False)

    raw_path = join(args.outdir, RAW_NETCDF_NAME)
    if exists(raw_path) and not args.keep_netcdf:
        remove(raw_path)

    print("Download complete.")
    print(f"Nearest ERA5 grid point: latitude={grid_lat}, longitude={grid_lon}")
    print(f"Records written: {len(precip_df)}")
    print(f"Time range: {precip_df.index.min()} to {precip_df.index.max()}")
    print(f"Output saved to {outpath}")
    if args.keep_netcdf and raw_netcdf_path and exists(raw_netcdf_path):
        print(f"Raw NetCDF retained at {raw_netcdf_path}")


if __name__ == "__main__":
    main()
