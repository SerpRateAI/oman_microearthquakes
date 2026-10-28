"""
Extract hub-event locations at preferred velocity scale factors for selected event groups.

Reads group metadata from ``event_group_info_selected_*.csv``, preferred scale factors from
``scale_factors_preferred.csv``, and best-fit locations from ``hub_event_location_info_group*.h5``
(written by ``localize_hub_event_3d_per_scale_factor.py``).
"""

from argparse import ArgumentParser
from pathlib import Path

from pandas import DataFrame, read_csv

from utils_basic import DETECTION_DIR, LOC_DIR, VEL_MODEL_DIR
from utils_loc import load_hub_event_location_from_hdf


DEFAULT_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3"
)


def load_preferred_scale_factors(filepath):
    df = read_csv(filepath)
    required = {"subarray", "scale_factor"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {filepath}; got {list(df.columns)}"
        )
    return {str(row["subarray"]): float(row["scale_factor"]) for _, row in df.iterrows()}


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Extract hub-event locations at preferred velocity scale factors "
            "for selected event groups"
        ),
    )
    parser.add_argument(
        "--group_info_csv",
        type=Path,
        default=Path(DETECTION_DIR)
        / f"event_group_info_selected_{DEFAULT_SUFFIX}.csv",
        help="CSV with selected event groups (columns include label, subarray, id_hub, …)",
    )
    parser.add_argument(
        "--scale_factors_preferred_csv",
        type=Path,
        default=Path(VEL_MODEL_DIR) / "scale_factors_preferred.csv",
        help="Preferred velocity scale factor per subarray (columns subarray, scale_factor)",
    )
    parser.add_argument(
        "--loc_dir",
        type=Path,
        default=Path(LOC_DIR),
        help="Directory containing hub_event_location_info_group*.h5 files",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path (default: LOC_DIR/hub_event_locations_preferred_scale_"
            "<group_info_stem>.csv)"
        ),
    )

    args = parser.parse_args()
    group_info_csv = args.group_info_csv
    scale_factors_preferred_csv = args.scale_factors_preferred_csv
    loc_dir = args.loc_dir

    if args.output_csv is None:
        output_csv = Path(LOC_DIR) / f"hub_event_locations_preferred_scale_{group_info_csv.stem}.csv"
    else:
        output_csv = args.output_csv

    print(f"Reading group info from {group_info_csv}...")
    group_df = read_csv(group_info_csv)

    print(f"Reading preferred scale factors from {scale_factors_preferred_csv}...")
    preferred_scale_by_subarray = load_preferred_scale_factors(scale_factors_preferred_csv)
    for subarray, sf in sorted(preferred_scale_by_subarray.items()):
        print(f"  Subarray {subarray}: scale factor {sf:.4f}")

    rows = []
    for _, group_row in group_df.iterrows():
        label = int(group_row["label"])
        subarray = str(group_row["subarray"])
        if subarray not in preferred_scale_by_subarray:
            raise ValueError(
                f"No preferred scale factor for subarray {subarray!r} "
                f"(group {label:d}); available: {sorted(preferred_scale_by_subarray)}"
            )

        scale_factor_preferred = preferred_scale_by_subarray[subarray]
        filepath_h5 = loc_dir / f"hub_event_location_info_group{label:d}.h5"
        if not filepath_h5.is_file():
            raise FileNotFoundError(
                f"Missing localization HDF5 for group {label:d}: {filepath_h5}"
            )

        location_dict, scale_factor_used, weight = load_hub_event_location_from_hdf(
            filepath_h5, scale_factor_preferred,
        )

        print(
            f"Group {label:d} (Subarray {subarray}, hub {int(group_row['id_hub']):d}): "
            f"scale {scale_factor_used:.4f}, "
            f"E={location_dict['east']:.2f} m, N={location_dict['north']:.2f} m, "
            f"D={location_dict['depth']:.2f} m, min misfit={location_dict['min_misfit']:.6f} s"
        )

        rows.append({
            **group_row.to_dict(),
            "scale_factor_preferred": scale_factor_preferred,
            "scale_factor_used": scale_factor_used,
            "weight": weight,
            "east": location_dict["east"],
            "north": location_dict["north"],
            "depth": location_dict["depth"],
            "min_misfit": location_dict["min_misfit"],
            "origin_time": location_dict["origin_time"].isoformat(),
        })

    out_df = DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df):d} locations to {output_csv}")
