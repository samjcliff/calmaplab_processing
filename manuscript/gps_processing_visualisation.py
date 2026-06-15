"""
GPS Processing Pipeline Visualisation
Figure dimensions: 7" x 5" (double column)

Creates a two-row figure showing:
  (a) Processing pipeline: Raw → Smoothed → Snapped GPS positions
  (b) Aggregation products: 30 m point, 30 m line segment, H3 hexagons

Each panel shows coloured markers/geometries on a CartoDB Positron basemap.
Simulated concentration values are added for demonstration if the data does
not already contain a 'concentration' column.

Expected input:
    A Parquet file of processed GPS data with columns:
        lat, lon             — raw positions
        lat_smooth, lon_smooth — Kalman-smoothed positions
        lat_snap, lon_snap   — road-snapped positions
        lat_30, lon_30       — 30 m grid-aggregated positions
        geometry             — WKB/WKT line segments for 30 m road links
        concentration        — (optional) measured pollutant value

Dependencies:
    Required:  pandas, numpy, matplotlib, geopandas, shapely, contextily,
               h3, pyproj

Usage:
    python manuscript/gps_processing_visualisation.py \\
        --gps-file data/figure_inputs/gps/processed_gps_2025-10-08_bbox.parquet \\
        --output-dir figures/
"""

import argparse
import os
import warnings
from pathlib import Path
import contextily as cx
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_INPUTS = REPO_ROOT / "data/figure_inputs"

# =============================================================================
# Configuration
# =============================================================================

# Bounding box for visualisation
BBOX = {
    "lat_min": 37.808,
    "lat_max": 37.813,
    "lon_min": -122.299,
    "lon_max": -122.292,
}

# Map tile provider
TILE_PROVIDER = cx.providers.CartoDB.PositronNoLabels


# =============================================================================
# AMT Journal Style
# =============================================================================

def setup_amt_style():
    """Configure matplotlib to match AMT journal requirements."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# =============================================================================
# Data Loading
# =============================================================================

def load_and_filter_data(filepath, bbox):
    """Load a GPS Parquet file and filter to the bounding box.

    Parameters
    ----------
    filepath : str or Path
    bbox : dict with keys lat_min, lat_max, lon_min, lon_max

    Returns
    -------
    DataFrame
    """
    df = pd.read_parquet(filepath)

    if "datetime" in df.columns and "sample_time" not in df.columns:
        df = df.rename(columns={"datetime": "sample_time"})

    mask = (
        (df["lat"] > bbox["lat_min"])
        & (df["lat"] < bbox["lat_max"])
        & (df["lon"] > bbox["lon_min"])
        & (df["lon"] < bbox["lon_max"])
    )
    return df[mask].copy()


def add_simulated_concentration(df, seed=42):
    """Add random concentration values for demonstration purposes."""
    rng = np.random.default_rng(seed)
    df["concentration"] = rng.uniform(1, 1000, size=len(df))
    return df


# =============================================================================
# GeoDataFrame Construction
# =============================================================================

def create_point_gdf(df, lon_col, lat_col, value_col="concentration"):
    """Aggregate points by unique location and return a GeoDataFrame."""
    valid = df[[lon_col, lat_col, value_col]].dropna()
    if len(valid) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
        crs="EPSG:4326",
    )
    gdf["geom_wkt"] = gdf.geometry.to_wkt()
    agg = gdf.groupby("geom_wkt").agg({value_col: "mean"}).reset_index()
    agg["geometry"] = gpd.GeoSeries.from_wkt(agg["geom_wkt"], crs="EPSG:4326")
    return gpd.GeoDataFrame(agg[[value_col, "geometry"]], crs="EPSG:4326")


def create_line_gdf(df, geometry_col="geometry", value_col="concentration"):
    """Parse a geometry column (WKT, WKB, or shapely) into a line GeoDataFrame."""
    if geometry_col not in df.columns:
        print(f"  Warning: geometry column '{geometry_col}' not found")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    valid = df[[geometry_col, value_col]].copy().dropna(subset=[geometry_col])

    # Drop empty strings / empty bytes
    def _is_valid(x):
        if x is None:
            return False
        if isinstance(x, str) and x.strip() == "":
            return False
        if isinstance(x, bytes) and len(x) == 0:
            return False
        return True

    valid = valid[valid[geometry_col].apply(_is_valid)]
    if len(valid) == 0:
        print(f"  Warning: No valid geometries in '{geometry_col}'")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    sample = valid[geometry_col].iloc[0]
    try:
        if isinstance(sample, bytes):
            geoms = gpd.GeoSeries.from_wkb(valid[geometry_col], crs="EPSG:4326")
        elif isinstance(sample, str):
            geoms = gpd.GeoSeries.from_wkt(valid[geometry_col], crs="EPSG:4326")
        elif hasattr(sample, "geom_type"):
            geoms = gpd.GeoSeries(valid[geometry_col].tolist(), crs="EPSG:4326")
        else:
            print(f"  Warning: Unknown geometry format: {type(sample)}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    except Exception as e:
        print(f"  Error parsing geometries: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        valid[[value_col]].reset_index(drop=True),
        geometry=geoms.reset_index(drop=True),
    )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    print(f"  Created {len(gdf)} line geometries")

    if len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf["geom_wkt"] = gdf.geometry.to_wkt()
    agg = gdf.groupby("geom_wkt").agg({value_col: "mean"}).reset_index()
    agg["geometry"] = gpd.GeoSeries.from_wkt(agg["geom_wkt"], crs="EPSG:4326")
    return gpd.GeoDataFrame(agg[[value_col, "geometry"]], crs="EPSG:4326")


def create_h3_hexagons(df, lon_col, lat_col, value_col="concentration", resolution=12):
    """Aggregate point data into H3 hexagonal cells."""
    valid = df[[lon_col, lat_col, value_col]].dropna()
    if len(valid) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    valid = valid.copy()
    valid["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(valid[lat_col], valid[lon_col])
    ]

    agg = valid.groupby("h3_index").agg({value_col: "mean"}).reset_index()

    def _h3_to_polygon(idx):
        boundary = h3.cell_to_boundary(idx)
        return Polygon([(lon, lat) for lat, lon in boundary])

    agg["geometry"] = agg["h3_index"].apply(_h3_to_polygon)
    return gpd.GeoDataFrame(agg[[value_col, "geometry"]], crs="EPSG:4326")


# =============================================================================
# Plotting Helpers
# =============================================================================

def plot_panel(ax, gdf, title, bbox, vmin, vmax,
               geom_type="point", point_size=1, line_width=1.5):
    """Plot a single map panel with basemap and data overlay."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(bbox["lon_min"], bbox["lat_min"])
    xmax, ymax = transformer.transform(bbox["lon_max"], bbox["lat_max"])

    if len(gdf) > 0:
        gdf_3857 = gdf.to_crs("EPSG:3857")
        plot_kwargs = dict(
            ax=ax, column="concentration", cmap="viridis",
            vmin=vmin, vmax=vmax, legend=False, zorder=2,
        )
        if geom_type == "point":
            gdf_3857.plot(**plot_kwargs, markersize=point_size)
        elif geom_type == "line":
            gdf_3857.plot(**plot_kwargs, linewidth=line_width)
        elif geom_type == "polygon":
            gdf_3857.plot(**plot_kwargs, edgecolor="none")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=TILE_PROVIDER, zoom=17, attribution=False)

    ax.set_title(title, fontsize=9, fontweight="normal", pad=3)
    ax.set_axis_off()


def draw_arrow(fig, ax_from, ax_to):
    """Draw a horizontal arrow between two axes in figure coordinates."""
    bbox_from = ax_from.get_position()
    bbox_to = ax_to.get_position()
    y_mid = (bbox_from.y0 + bbox_from.y1) / 2

    arrow = FancyArrowPatch(
        (bbox_from.x1 + 0.005, y_mid),
        (bbox_to.x0 - 0.005, y_mid),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=10,
        color="black",
        linewidth=1,
    )
    fig.patches.append(arrow)


# =============================================================================
# Main Figure
# =============================================================================

def make_figure(gps_data, output_dir):
    """Build the two-row GPS processing pipeline figure and save."""
    setup_amt_style()

    # Build data layers
    print("Creating data layers...")
    raw_points = create_point_gdf(gps_data, "lon", "lat")
    smoothed_points = create_point_gdf(gps_data, "lon_smooth", "lat_smooth")
    snapped_points = create_point_gdf(gps_data, "lon_snap", "lat_snap")
    segment_points = create_point_gdf(gps_data, "lon_30", "lat_30")
    segment_lines = create_line_gdf(gps_data, "geometry")
    h3_hexagons = create_h3_hexagons(gps_data, "lon_snap", "lat_snap", resolution=12)

    vmin = gps_data["concentration"].min()
    vmax = gps_data["concentration"].max()
    print(f"Concentration range: {vmin:.1f} – {vmax:.1f}")

    # Layout
    fig = plt.figure(figsize=(7, 5), dpi=150)

    row_a_bottom, row_a_height = 0.52, 0.40
    row_b_bottom, row_b_height = 0.08, 0.40

    panel_w = 0.28
    arrow_sp = 0.04
    row_a_left = (1 - 3 * panel_w - 2 * arrow_sp) / 2

    ax_raw = fig.add_axes([row_a_left, row_a_bottom, panel_w, row_a_height])
    ax_smooth = fig.add_axes([row_a_left + panel_w + arrow_sp, row_a_bottom, panel_w, row_a_height])
    ax_snap = fig.add_axes([row_a_left + 2 * (panel_w + arrow_sp), row_a_bottom, panel_w, row_a_height])

    row_b_sp = 0.03
    row_b_left = (1 - 3 * panel_w - 2 * row_b_sp) / 2

    ax_pt = fig.add_axes([row_b_left, row_b_bottom, panel_w, row_b_height])
    ax_ln = fig.add_axes([row_b_left + panel_w + row_b_sp, row_b_bottom, panel_w, row_b_height])
    ax_h3 = fig.add_axes([row_b_left + 2 * (panel_w + row_b_sp), row_b_bottom, panel_w, row_b_height])

    # Row (a)
    print("Plotting row (a): Processing pipeline...")
    plot_panel(ax_raw, raw_points, "Raw", BBOX, vmin, vmax, "point", point_size=1)
    plot_panel(ax_smooth, smoothed_points, "Smoothed", BBOX, vmin, vmax, "point", point_size=1)
    plot_panel(ax_snap, snapped_points, "Snapped", BBOX, vmin, vmax, "point", point_size=1)
    draw_arrow(fig, ax_raw, ax_smooth)
    draw_arrow(fig, ax_smooth, ax_snap)

    # Row (b)
    print("Plotting row (b): Aggregation products...")
    plot_panel(ax_pt, segment_points, "30 m point", BBOX, vmin, vmax, "point", point_size=1)
    plot_panel(ax_ln, segment_lines, "30 m line segment", BBOX, vmin, vmax, "line", line_width=1.5)
    plot_panel(ax_h3, h3_hexagons, "H3 index (R=12)", BBOX, vmin, vmax, "polygon")

    # Row labels
    fig.text(0.02, row_a_bottom + row_a_height + 0.02, "(a)",
             fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.06, row_a_bottom + row_a_height + 0.02, "Processing pipeline",
             fontsize=9, va="bottom")
    fig.text(0.02, row_b_bottom + row_b_height + 0.02, "(b)",
             fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.06, row_b_bottom + row_b_height + 0.02,
             "Aggregation products by drive pass or time period",
             fontsize=9, va="bottom")

    # Scalebar (200 m)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin_m, _ = transformer.transform(BBOX["lon_min"], BBOX["lat_min"])
    xmax_m, _ = transformer.transform(BBOX["lon_max"], BBOX["lat_max"])
    panel_width_m = xmax_m - xmin_m

    ax_pt_pos = ax_pt.get_position()
    scalebar_y = 0.04
    scalebar_x = ax_pt_pos.x0
    scalebar_len_m = 200
    scalebar_w = (scalebar_len_m / panel_width_m) * ax_pt_pos.width

    fig.add_artist(Line2D(
        [scalebar_x, scalebar_x + scalebar_w], [scalebar_y, scalebar_y],
        transform=fig.transFigure, color="black", linewidth=2, solid_capstyle="butt",
    ))
    tick_h = 0.008
    for x in [scalebar_x, scalebar_x + scalebar_w]:
        fig.add_artist(Line2D(
            [x, x], [scalebar_y - tick_h / 2, scalebar_y + tick_h / 2],
            transform=fig.transFigure, color="black", linewidth=1.5,
        ))
    fig.text(scalebar_x + scalebar_w / 2, scalebar_y + 0.012,
             f"{scalebar_len_m} m", ha="center", va="bottom", fontsize=8)

    fig.text(0.98, scalebar_y, "© OpenStreetMap contributors © CARTO",
             fontsize=7, color="grey", ha="right", va="center")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = os.path.join(output_dir, f"gps_processing_pipeline_vis.{fmt}")
        print(f"Saving {fmt.upper()}: {path}")
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)
    print("Done!")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the GPS processing pipeline figure (AMT style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gps-file",
        default=str(FIGURE_INPUTS / "gps/processed_gps_2025-10-08_bbox.parquet"),
        help="Path to processed GPS Parquet file "
             "(default: data/figure_inputs/gps/processed_gps_2025-10-08_bbox.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "figures"),
        help="Output directory (default: figures/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.gps_file):
        raise FileNotFoundError(
            f"GPS file not found: {args.gps_file}\n"
            f"Provide the correct path with: --gps-file /path/to/file.parquet"
        )

    print(f"Loading: {args.gps_file}")
    gps_data = load_and_filter_data(args.gps_file, BBOX)
    print(f"  {len(gps_data):,} points within bounding box")

    if len(gps_data) == 0:
        raise ValueError(
            "No data points within bounding box. "
            "Adjust BBOX in the script to match your data extent."
        )

    if "concentration" not in gps_data.columns:
        print("  Adding simulated concentration values for demonstration...")
        gps_data = add_simulated_concentration(gps_data)

    make_figure(gps_data, args.output_dir)


if __name__ == "__main__":
    main()