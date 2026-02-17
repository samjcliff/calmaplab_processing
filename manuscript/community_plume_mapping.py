"""
West Oakland WWTP Combined Figure
AMT Journal Style - (a) single map + timeseries, (b) 2x2 species grid

Produces a two-part figure for the CalMAPLab manuscript detailing:
  (a) CH4 spatial map and corresponding timeseries for a wastewater treatment plant plume chasing drive
  (b) Row of three maps showing CH4, C2H6, and CH5S from a full West Oakland drive

Expected database schema (DuckDB):
    Table: measurements
    Columns:
        sample_time  (TIMESTAMP WITH TIME ZONE)
        parameter    (VARCHAR) - species identifier, e.g. 'CH4', 'C2H6', 'CH5S'
        value        (DOUBLE)  - measured concentration
        drive_pass   (VARCHAR) - identifier for individual drive passes
        lat_30       (DOUBLE)  - latitude, snapped to ~30 m grid
        lon_30       (DOUBLE)  - longitude, snapped to ~30 m grid
        rd_type      (VARCHAR) - road classification
        summary_flag (VARCHAR) - QA/QC flag

Dependencies:
    Required:  duckdb, pandas, matplotlib
    Optional:  contextily (basemap tiles), geopandas + shapely (reprojection),
               matplotlib-scalebar (scale bars on maps)

Usage:
    python community_plume_mapping.py --db data/drives_r2.duckdb --output-dir figures/

    All paths default to sensible relative locations; override with flags as needed.
"""

import argparse
import os
import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Optional dependencies - degrade gracefully
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False


# =============================================================================
# Configuration
# =============================================================================

# West Oakland bounding box
BBOX = {
    "lat_min": 37.794,
    "lat_max": 37.830,
    "lon_min": -122.330,
    "lon_max": -122.262,
}

# Part (a) - single-session drive
PART_A = {
    "date_start": "2025-02-25",
    "date_end": "2025-02-26",
    "time_start": "21:00:00",
    "time_end": "00:39:00",
    "species": "CH4",
}

# Part (b) - multi-species comparison
PART_B = {
    "date": "2025-05-07",
    "species": ["CH4", "C2H6", "CH5S"],
}

# Species display metadata
SPECIES_INFO = {
    "CH4":  {"label": r"CH$_4$",      "unit": "ppm"},
    "C2H6": {"label": r"C$_2$H$_6$",  "unit": "ppb"},
    "CH5S": {"label": r"CH$_5$S$^+$", "unit": "ppb"},
}

# Plot appearance
STYLE = {
    "cmap": "viridis",
    "point_size": 3,
    "point_alpha": 0.9,
    "line_color": "#404788",
    "line_width": 0.5,
}


# =============================================================================
# AMT Journal Style
# =============================================================================

def setup_amt_style():
    """Configure matplotlib to match AMT journal requirements."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =============================================================================
# Data Processing
# =============================================================================

def process_for_map(df, parameter):
    """Aggregate measurement data onto the spatial grid for mapping.

    Steps:
        1. Mean value per grid cell per drive pass
        2. Median across passes per grid cell
        3. Cap at 5th/95th percentiles to reduce outlier influence

    Returns a DataFrame with columns: lat_30, lon_30, value, value_capped, n_passes
    """
    sub = df[df["parameter"] == parameter].copy()
    if len(sub) == 0:
        return None

    # Per-pass mean, then cross-pass median
    agg_pass = (
        sub.groupby(["lat_30", "lon_30", "drive_pass"])
        .agg(value=("value", "mean"))
        .reset_index()
    )
    agg_grid = (
        agg_pass.groupby(["lat_30", "lon_30"])
        .agg(value=("value", "median"), n_passes=("value", "count"))
        .reset_index()
    )

    lower = agg_grid["value"].quantile(0.05)
    upper = agg_grid["value"].quantile(0.95)
    agg_grid["value_capped"] = agg_grid["value"].clip(lower=lower, upper=upper)

    return agg_grid


def process_for_timeseries(df, parameter):
    """Extract a time-indexed Series of raw measurements for one species."""
    sub = df[df["parameter"] == parameter].copy()
    if len(sub) == 0:
        return None

    sub["sample_time"] = pd.to_datetime(sub["sample_time"])
    sub = sub.sort_values("sample_time").set_index("sample_time")
    return sub["value"]


# =============================================================================
# Plotting Helpers
# =============================================================================

def create_map_panel(ax, map_data, add_scalebar=False):
    """Plot spatial data on a map axis with optional basemap tiles.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    map_data : DataFrame from process_for_map(), or None
    add_scalebar : bool

    Returns
    -------
    scatter : PathCollection or None
    """
    scatter = None

    if map_data is not None and len(map_data) > 0:
        if HAS_GEOPANDAS:
            geometry = [
                Point(lon, lat)
                for lon, lat in zip(map_data["lon_30"], map_data["lat_30"])
            ]
            gdf = gpd.GeoDataFrame(map_data, geometry=geometry, crs="EPSG:4326")
            gdf = gdf.to_crs(epsg=3857)

            if HAS_CONTEXTILY:
                bounds = gdf.total_bounds
                buffer = 100
                ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
                ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
                try:
                    ctx.add_basemap(
                        ax,
                        source=ctx.providers.CartoDB.PositronNoLabels,
                        zoom=14,
                        attribution=False,
                    )
                except Exception as e:
                    print(f"    Warning: Could not fetch basemap tiles: {e}")

            scatter = ax.scatter(
                gdf.geometry.x,
                gdf.geometry.y,
                c=gdf["value_capped"],
                cmap=STYLE["cmap"],
                s=STYLE["point_size"],
                alpha=STYLE["point_alpha"],
                zorder=5,
                linewidths=0,
            )

            if HAS_SCALEBAR and add_scalebar:
                scalebar = ScaleBar(
                    1,
                    location="lower left",
                    length_fraction=0.25,
                    font_properties={"size": 7},
                    box_alpha=0.7,
                    sep=2,
                    pad=0.3,
                )
                ax.add_artist(scalebar)
        else:
            # Fallback without geopandas: plot in lat/lon directly
            scatter = ax.scatter(
                map_data["lon_30"],
                map_data["lat_30"],
                c=map_data["value_capped"],
                cmap=STYLE["cmap"],
                s=STYLE["point_size"],
                alpha=STYLE["point_alpha"],
                linewidths=0,
            )
            ax.set_xlim(BBOX["lon_min"], BBOX["lon_max"])
            ax.set_ylim(BBOX["lat_min"], BBOX["lat_max"])

        ax.set_aspect("equal")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)

    # Clean map axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return scatter


# =============================================================================
# Database Queries
# =============================================================================

def query_data(db_path):
    """Run all queries against the DuckDB database and return three DataFrames.

    Returns
    -------
    df_map_a : DataFrame  - Part (a) spatial data
    df_ts_a  : DataFrame  - Part (a) timeseries data
    df_map_b : DataFrame  - Part (b) spatial data for all species
    """
    con = duckdb.connect(db_path, read_only=True)

    # Part (a) - map data
    query_map_a = """
    SELECT sample_time, parameter, value, drive_pass, lat_30, lon_30, rd_type, summary_flag
    FROM measurements
    WHERE parameter = $1
      AND sample_time >= ($2 || ' ' || $3 || '-00:00')::TIMESTAMPTZ
      AND sample_time <= ($4 || ' ' || $5 || '-00:00')::TIMESTAMPTZ
      AND lat_30 BETWEEN $6 AND $7
      AND lon_30 BETWEEN $8 AND $9
    """
    df_map_a = con.execute(
        query_map_a,
        [
            PART_A["species"],
            PART_A["date_start"], PART_A["time_start"],
            PART_A["date_end"], PART_A["time_end"],
            BBOX["lat_min"], BBOX["lat_max"],
            BBOX["lon_min"], BBOX["lon_max"],
        ],
    ).fetchdf()
    print(f"  Part (a) map data: {len(df_map_a):,} rows")

    # Part (a) - timeseries (full drive, not bbox-restricted)
    query_ts_a = """
    SELECT sample_time, parameter, value, summary_flag
    FROM measurements
    WHERE parameter = $1
      AND sample_time >= ($2 || ' ' || $3 || '-00:00')::TIMESTAMPTZ
      AND sample_time <= ($4 || ' ' || $5 || '-00:00')::TIMESTAMPTZ
    """
    df_ts_a = con.execute(
        query_ts_a,
        [
            PART_A["species"],
            PART_A["date_start"], PART_A["time_start"],
            PART_A["date_end"], PART_A["time_end"],
        ],
    ).fetchdf()
    print(f"  Part (a) timeseries: {len(df_ts_a):,} rows")

    # Part (b) - multi-species maps
    # DuckDB supports list parameters for IN clauses via unnest
    query_map_b = """
    SELECT sample_time, parameter, value, drive_pass, lat_30, lon_30, rd_type, summary_flag
    FROM measurements
    WHERE parameter IN (SELECT unnest($1::VARCHAR[]))
      AND sample_time >= ($2 || ' 00:00:00')::TIMESTAMPTZ
      AND sample_time <  ($2 || ' 23:59:59')::TIMESTAMPTZ
      AND lat_30 BETWEEN $3 AND $4
      AND lon_30 BETWEEN $5 AND $6
    """
    df_map_b = con.execute(
        query_map_b,
        [
            PART_B["species"],
            PART_B["date"],
            BBOX["lat_min"], BBOX["lat_max"],
            BBOX["lon_min"], BBOX["lon_max"],
        ],
    ).fetchdf()
    print(f"  Part (b) map data: {len(df_map_b):,} rows")
    print(f"    Species found: {df_map_b['parameter'].unique().tolist()}")

    con.close()
    return df_map_a, df_ts_a, df_map_b


# =============================================================================
# Main Figure
# =============================================================================

def make_figure(df_map_a, df_ts_a, df_map_b, output_dir):
    """Build and save the combined two-part figure."""

    setup_amt_style()

    fig = plt.figure(figsize=(7, 5.5))

    # ---- Layout geometry ----
    left_margin = 0.07
    right_margin = 0.02
    h_gap = 0.04

    total_width = 1 - left_margin - right_margin
    map_width = (total_width - 2 * h_gap) / 3
    ts_width = 2 * map_width + h_gap

    cbar_height = 0.018
    cbar_pad = 0.008

    # ---- Part (a): map + timeseries ----
    a_top = 0.98
    a_map_height = 0.28

    a_map_left = left_margin
    a_map_bottom = a_top - a_map_height

    a_cbar_bottom = a_map_bottom - cbar_pad - cbar_height
    a_cbar_left = a_map_left + 0.01
    a_cbar_width = map_width - 0.02

    b_plot3_right = left_margin + 2 * (map_width + h_gap) + map_width
    a_ts_left = left_margin + map_width + h_gap + 0.06
    a_ts_width = b_plot3_right - a_ts_left - 0.01
    a_ts_bottom = a_map_bottom + 0.02
    a_ts_height = a_map_height - 0.02

    ax_map_a = fig.add_axes([a_map_left, a_map_bottom, map_width, a_map_height])
    ax_cbar_a = fig.add_axes([a_cbar_left, a_cbar_bottom, a_cbar_width, cbar_height])
    ax_ts_a = fig.add_axes([a_ts_left, a_ts_bottom, a_ts_width, a_ts_height])

    # Map (a)
    map_data_a = process_for_map(df_map_a, PART_A["species"])
    scatter_a = create_map_panel(ax_map_a, map_data_a, add_scalebar=True)

    if scatter_a is not None:
        info = SPECIES_INFO[PART_A["species"]]
        cbar_a = plt.colorbar(scatter_a, cax=ax_cbar_a, orientation="horizontal")
        cbar_a.ax.tick_params(labelsize=7, width=0.5, length=2)
        cbar_a.outline.set_linewidth(0.5)
        cbar_a.set_label(f"{info['label']} ({info['unit']})", fontsize=8)

    # Timeseries (a)
    ts_a = process_for_timeseries(df_ts_a, PART_A["species"])
    if ts_a is not None and len(ts_a) > 0:
        ax_ts_a.plot(ts_a.index, ts_a.values,
                     linewidth=STYLE["line_width"], color=STYLE["line_color"])
        ax_ts_a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax_ts_a.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax_ts_a.tick_params(axis="x", labelsize=7)
        ax_ts_a.tick_params(axis="y", labelsize=7)
        info = SPECIES_INFO[PART_A["species"]]
        ax_ts_a.set_xlabel("Time (UTC)", fontsize=8)
        ax_ts_a.set_ylabel(f"{info['label']} ({info['unit']})", fontsize=8)
        ax_ts_a.grid(True, alpha=0.3, linewidth=0.3, color="grey")
        ax_ts_a.set_axisbelow(True)

    ax_ts_a.spines["top"].set_visible(False)
    ax_ts_a.spines["right"].set_visible(False)
    ax_ts_a.spines["left"].set_linewidth(0.5)
    ax_ts_a.spines["bottom"].set_linewidth(0.5)

    fig.text(0.01, a_top - 0.005, "(a)", fontsize=9, fontweight="bold",
             va="top", ha="left")

    # ---- Part (b): three species maps ----
    b_top = a_cbar_bottom - 0.10
    b_map_height = a_map_height

    for idx, species in enumerate(PART_B["species"]):
        panel_left = left_margin + idx * (map_width + h_gap)
        panel_bottom = b_top - b_map_height

        cb_bottom = panel_bottom - cbar_pad - cbar_height
        cb_left = panel_left + 0.01
        cb_width = map_width - 0.02

        ax_map = fig.add_axes([panel_left, panel_bottom, map_width, b_map_height])
        ax_cbar = fig.add_axes([cb_left, cb_bottom, cb_width, cbar_height])

        map_data = process_for_map(df_map_b, species)
        scatter = create_map_panel(ax_map, map_data, add_scalebar=True)

        if scatter is not None:
            info = SPECIES_INFO[species]
            cbar = plt.colorbar(scatter, cax=ax_cbar, orientation="horizontal")
            cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
            cbar.outline.set_linewidth(0.5)
            cbar.set_label(f"{info['label']} ({info['unit']})", fontsize=8)
        else:
            ax_cbar.set_visible(False)

    fig.text(0.01, b_top - 0.005, "(b)", fontsize=9, fontweight="bold",
             va="top", ha="left")

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, "wo_wwtp_combined_amt.pdf")
    png_path = os.path.join(output_dir, "wo_wwtp_combined_amt.png")

    print(f"Saving PDF: {pdf_path}")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saving PNG: {png_path}")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")

    plt.close()
    print("Done!")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the West Oakland WWTP combined figure (AMT style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default="data/drives_r2.duckdb",
        help="Path to the DuckDB database (default: data/drives_r2.duckdb)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/",
        help="Directory for output files (default: figures/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.db):
        raise FileNotFoundError(
            f"Database not found: {args.db}\n"
            f"Provide the correct path with: python {os.path.basename(__file__)} --db /path/to/drives_r2.duckdb"
        )

    missing = []
    if not HAS_CONTEXTILY:
        missing.append("contextily (basemap tiles will be missing)")
    if not HAS_GEOPANDAS:
        missing.append("geopandas (maps will use unprojected lat/lon)")
    if not HAS_SCALEBAR:
        missing.append("matplotlib-scalebar (scale bars will be missing)")
    if missing:
        print("Optional dependencies not found:")
        for m in missing:
            print(f"  - {m}")
        print()

    print(f"Database: {args.db}")
    print("Querying data...")
    df_map_a, df_ts_a, df_map_b = query_data(args.db)

    print("Building figure...")
    make_figure(df_map_a, df_ts_a, df_map_b, args.output_dir)


if __name__ == "__main__":
    main()