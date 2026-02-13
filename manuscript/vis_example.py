#!/usr/bin/env python3
"""
Visualizing GPS Processing Pipeline

Creates a two-panel figure showing:
  a) GPS processing pipeline: Raw → Smoothed → Snapped
  b) Aggregation products: 30m point, 30m line segment, H3 hexagons

Requirements:
  - Processed GPS data (Parquet format with complete output)
  - contextily, h3, shapely packages for visualization
"""

import warnings
from pathlib import Path

import contextily as cx
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================

GPS_DATA_PATH = Path(
    "F:/calmaplab_processing/data/processed/gps/processed_gps_2025-10-08.parquet"
)
OUTPUT_DIR = Path("F:/calmaplab_processing/figures/")

#Bounding box for visualization (adjust to your study area)
BBOX = {
    "lat_min": 37.808,
    "lat_max": 37.813,
    "lon_min": -122.299,
    "lon_max": -122.292,
}

# Map tile provider (CartoDB Positron without labels)
TILE_PROVIDER = cx.providers.CartoDB.PositronNoLabels

# Color scale limits (will be set from data if None)
CONCENTRATION_RANGE = None


# =============================================================================
# Data loading and preparation
# =============================================================================


def load_and_filter_data(filepath: Path, bbox: dict) -> pd.DataFrame:
    """Load parquet data and filter to bounding box."""
    df = pd.read_parquet(filepath)

    # Rename datetime column if needed
    if "datetime" in df.columns and "sample_time" not in df.columns:
        df = df.rename(columns={"datetime": "sample_time"})

    # Filter to bounding box
    mask = (
        (df["lat"] > bbox["lat_min"])
        & (df["lat"] < bbox["lat_max"])
        & (df["lon"] > bbox["lon_min"])
        & (df["lon"] < bbox["lon_max"])
    )
    df = df[mask].copy()

    return df


def add_simulated_concentration(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add simulated pollutant values for demonstration."""
    rng = np.random.default_rng(seed)
    df["concentration"] = rng.uniform(1, 1000, size=len(df))
    return df


# =============================================================================
# Create GeoDataFrames for each processing stage
# =============================================================================


def create_point_gdf(
    df: pd.DataFrame, lon_col: str, lat_col: str, value_col: str = "concentration"
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame of points, aggregating by unique location."""
    valid = df[[lon_col, lat_col, value_col]].dropna()
    if len(valid) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
        crs="EPSG:4326",
    )

    # Aggregate by geometry (mean concentration per unique point)
    gdf["geom_wkt"] = gdf.geometry.to_wkt()
    agg = gdf.groupby("geom_wkt").agg({value_col: "mean"}).reset_index()
    agg["geometry"] = gpd.GeoSeries.from_wkt(agg["geom_wkt"], crs="EPSG:4326")
    return gpd.GeoDataFrame(agg[[value_col, "geometry"]], crs="EPSG:4326")


def create_line_gdf(
    df: pd.DataFrame, geometry_col: str = "geometry", value_col: str = "concentration"
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame of line segments from geometry column (WKT, WKB, or shapely)."""
    import shapely
    
    if geometry_col not in df.columns:
        print(f"Warning: geometry column '{geometry_col}' not found in dataframe")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Start with a copy and filter nulls
    valid = df[[geometry_col, value_col]].copy()
    valid = valid.dropna(subset=[geometry_col])
    
    # Also filter empty strings and empty bytes
    def is_valid_geom_value(x):
        if x is None:
            return False
        if isinstance(x, str) and (x == "" or x.strip() == ""):
            return False
        if isinstance(x, bytes) and len(x) == 0:
            return False
        return True
    
    valid = valid[valid[geometry_col].apply(is_valid_geom_value)]
    
    if len(valid) == 0:
        print(f"Warning: No valid geometries found in '{geometry_col}'")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Detect geometry format and parse accordingly
    sample = valid[geometry_col].iloc[0]
    print(f"Geometry column type: {type(sample)}, sample length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
    
    try:
        if isinstance(sample, bytes):
            # WKB format (binary)
            geoms = gpd.GeoSeries.from_wkb(valid[geometry_col], crs="EPSG:4326")
        elif isinstance(sample, str):
            # WKT format (string)
            geoms = gpd.GeoSeries.from_wkt(valid[geometry_col], crs="EPSG:4326")
        elif hasattr(sample, "geom_type"):
            # Already shapely geometry objects
            geoms = gpd.GeoSeries(valid[geometry_col].tolist(), crs="EPSG:4326")
        else:
            print(f"Warning: Unknown geometry format: {type(sample)}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    except Exception as e:
        print(f"Error parsing geometries: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(valid[[value_col]].reset_index(drop=True), geometry=geoms.reset_index(drop=True))
    
    # Filter out any invalid/empty geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    
    print(f"Created {len(gdf)} line geometries")
    
    if len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Aggregate by geometry
    gdf["geom_wkt"] = gdf.geometry.to_wkt()
    agg = gdf.groupby("geom_wkt").agg({value_col: "mean"}).reset_index()
    agg["geometry"] = gpd.GeoSeries.from_wkt(agg["geom_wkt"], crs="EPSG:4326")
    return gpd.GeoDataFrame(agg[[value_col, "geometry"]], crs="EPSG:4326")


def create_h3_hexagons(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    value_col: str = "concentration",
    resolution: int = 12,
) -> gpd.GeoDataFrame:
    """Create H3 hexagonal aggregation from point data."""
    valid = df[[lon_col, lat_col, value_col]].dropna()
    if len(valid) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Get H3 indices for each point
    valid = valid.copy()
    valid["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(valid[lat_col], valid[lon_col])
    ]

    # Aggregate by H3 cell
    agg = valid.groupby("h3_index").agg({value_col: "mean"}).reset_index()

    # Convert H3 indices to polygons
    def h3_to_polygon(h3_index):
        boundary = h3.cell_to_boundary(h3_index)
        # h3 returns (lat, lon) pairs, need to swap to (lon, lat) for shapely
        coords = [(lon, lat) for lat, lon in boundary]
        return Polygon(coords)

    agg["geometry"] = agg["h3_index"].apply(h3_to_polygon)
    return gpd.GeoDataFrame(agg[[value_col, "geometry"]], crs="EPSG:4326")


# =============================================================================
# Plotting functions
# =============================================================================


def plot_panel(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    title: str,
    bbox: dict,
    vmin: float,
    vmax: float,
    geom_type: str = "point",
    point_size: float = 1,
    line_width: float = 1.5,
) -> None:
    """Plot a single map panel with basemap and data overlay."""
    # Set axis limits first (in Web Mercator for contextily)
    ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
    ax.set_ylim(bbox["lat_min"], bbox["lat_max"])

    # Convert to Web Mercator for plotting with contextily
    if len(gdf) > 0:
        gdf_3857 = gdf.to_crs("EPSG:3857")

        if geom_type == "point":
            gdf_3857.plot(
                ax=ax,
                column="concentration",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                markersize=point_size,
                legend=False,
                zorder=2,
            )
        elif geom_type == "line":
            gdf_3857.plot(
                ax=ax,
                column="concentration",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                linewidth=line_width,
                legend=False,
                zorder=2,
            )
        elif geom_type == "polygon":
            gdf_3857.plot(
                ax=ax,
                column="concentration",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                edgecolor="none",
                legend=False,
                zorder=2,
            )

    # Transform axis limits to Web Mercator
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(bbox["lon_min"], bbox["lat_min"])
    xmax, ymax = transformer.transform(bbox["lon_max"], bbox["lat_max"])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add basemap (attribution disabled - added once at figure level)
    cx.add_basemap(ax, source=TILE_PROVIDER, zoom=17, attribution=False)

    # Style
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_axis_off()


def draw_arrow(fig, ax_from, ax_to):
    """Draw an arrow between two axes."""
    # Get positions in figure coordinates
    bbox_from = ax_from.get_position()
    bbox_to = ax_to.get_position()

    # Arrow from right edge of ax_from to left edge of ax_to
    x_start = bbox_from.x1
    x_end = bbox_to.x0
    y_mid = (bbox_from.y0 + bbox_from.y1) / 2

    arrow = FancyArrowPatch(
        (x_start + 0.005, y_mid),
        (x_end - 0.005, y_mid),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=10,
        color="black",
        linewidth=1,
    )
    fig.patches.append(arrow)


# =============================================================================
# Main figure creation
# =============================================================================


def create_figure(gps_data: pd.DataFrame, output_dir: Path) -> None:
    """Create the complete two-row visualization figure."""

    # Create data layers
    print("Creating data layers...")

    raw_points = create_point_gdf(gps_data, "lon", "lat")
    smoothed_points = create_point_gdf(gps_data, "lon_smooth", "lat_smooth")
    snapped_points = create_point_gdf(gps_data, "lon_snap", "lat_snap")
    segment_points = create_point_gdf(gps_data, "lon_30", "lat_30")
    segment_lines = create_line_gdf(gps_data, "geometry")
    h3_hexagons = create_h3_hexagons(gps_data, "lon_snap", "lat_snap", resolution=12)

    # Determine concentration range
    global CONCENTRATION_RANGE
    if CONCENTRATION_RANGE is None:
        CONCENTRATION_RANGE = (
            gps_data["concentration"].min(),
            gps_data["concentration"].max(),
        )
    vmin, vmax = CONCENTRATION_RANGE

    print(f"Concentration range: {vmin:.1f} - {vmax:.1f}")

    # Create figure with custom layout
    fig = plt.figure(figsize=(7, 5), dpi=150)

    # Row heights and spacing
    row_a_bottom = 0.52
    row_a_height = 0.40
    row_b_bottom = 0.08
    row_b_height = 0.40
    label_height = 0.04

    # Panel widths for row a (3 panels + 2 arrow spaces)
    panel_width = 0.28
    arrow_space = 0.04
    row_a_total = 3 * panel_width + 2 * arrow_space
    row_a_left = (1 - row_a_total) / 2

    # Row a: Processing pipeline
    ax_raw = fig.add_axes(
        [row_a_left, row_a_bottom, panel_width, row_a_height]
    )
    ax_smoothed = fig.add_axes(
        [row_a_left + panel_width + arrow_space, row_a_bottom, panel_width, row_a_height]
    )
    ax_snapped = fig.add_axes(
        [row_a_left + 2 * (panel_width + arrow_space), row_a_bottom, panel_width, row_a_height]
    )

    # Row b: Aggregation products (3 equal panels)
    row_b_panel_width = 0.28
    row_b_spacing = 0.03
    row_b_total = 3 * row_b_panel_width + 2 * row_b_spacing
    row_b_left = (1 - row_b_total) / 2

    ax_seg_point = fig.add_axes(
        [row_b_left, row_b_bottom, row_b_panel_width, row_b_height]
    )
    ax_seg_line = fig.add_axes(
        [row_b_left + row_b_panel_width + row_b_spacing, row_b_bottom, row_b_panel_width, row_b_height]
    )
    ax_h3 = fig.add_axes(
        [row_b_left + 2 * (row_b_panel_width + row_b_spacing), row_b_bottom, row_b_panel_width, row_b_height]
    )

    # Plot row a panels
    print("Plotting row a: Processing pipeline...")
    plot_panel(ax_raw, raw_points, "Raw", BBOX, vmin, vmax, "point", point_size=1)
    plot_panel(ax_smoothed, smoothed_points, "Smoothed", BBOX, vmin, vmax, "point", point_size=1)
    plot_panel(ax_snapped, snapped_points, "Snapped", BBOX, vmin, vmax, "point", point_size=1)

    # Draw arrows between row a panels
    draw_arrow(fig, ax_raw, ax_smoothed)
    draw_arrow(fig, ax_smoothed, ax_snapped)

    # Plot row b panels
    print("Plotting row b: Aggregation products...")
    plot_panel(
        ax_seg_point, segment_points, "30 m point", BBOX, vmin, vmax,
        "point", point_size=1
    )
    plot_panel(
        ax_seg_line, segment_lines, "30 m line segment", BBOX, vmin, vmax,
        "line", line_width=1.5
    )
    plot_panel(
        ax_h3, h3_hexagons, "H3 index (R=12)", BBOX, vmin, vmax, "polygon"
    )

    # Add row labels
    fig.text(
        0.02, row_a_bottom + row_a_height + 0.02,
        "a)   Processing pipeline",
        fontsize=11, fontweight="bold", va="bottom"
    )
    fig.text(
        0.02, row_b_bottom + row_b_height + 0.02,
        "b)   Aggregation products by drive pass or time period",
        fontsize=11, fontweight="bold", va="bottom"
    )

    # Add scalebar below the first panel of row b (200 m)
    scalebar_y = 0.04
    ax_seg_point_pos = ax_seg_point.get_position()
    scalebar_x_start = ax_seg_point_pos.x0
    
    # Calculate scalebar width in figure coordinates
    # Get the panel width in meters from the Web Mercator extent
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin_m, _ = transformer.transform(BBOX["lon_min"], BBOX["lat_min"])
    xmax_m, _ = transformer.transform(BBOX["lon_max"], BBOX["lat_max"])
    panel_width_m = xmax_m - xmin_m
    panel_width_fig = ax_seg_point_pos.width
    scalebar_length_m = 200
    scalebar_width_fig = (scalebar_length_m / panel_width_m) * panel_width_fig
    
    # Draw scalebar line
    from matplotlib.lines import Line2D
    scalebar_line = Line2D(
        [scalebar_x_start, scalebar_x_start + scalebar_width_fig],
        [scalebar_y, scalebar_y],
        transform=fig.transFigure,
        color="black",
        linewidth=2,
        solid_capstyle="butt"
    )
    fig.add_artist(scalebar_line)
    
    # Add ticks at ends
    tick_height = 0.008
    for x in [scalebar_x_start, scalebar_x_start + scalebar_width_fig]:
        tick = Line2D(
            [x, x],
            [scalebar_y - tick_height/2, scalebar_y + tick_height/2],
            transform=fig.transFigure,
            color="black",
            linewidth=1.5
        )
        fig.add_artist(tick)
    
    # Add label
    fig.text(
        scalebar_x_start + scalebar_width_fig / 2,
        scalebar_y + 0.012,
        f"{scalebar_length_m} m",
        ha="center",
        va="bottom",
        fontsize=8
    )

    # Add attribution (aligned with scalebar)
    fig.text(
        0.98, scalebar_y,
        "© OpenStreetMap contributors © CARTO",
        fontsize=7, color="grey", ha="right", va="center"
    )

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "gps_processing_pipeline_vis.pdf"
    png_path = output_dir / "gps_processing_pipeline_vis.png"

    print(f"Saving to {pdf_path}...")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.1)

    print(f"Saving to {png_path}...")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)
    print(f"Figure saved to: {output_dir}")


# =============================================================================
# Main entry point
# =============================================================================


def main():
    """Main function to run the visualization pipeline."""
    print(f"Loading data from: {GPS_DATA_PATH}")

    if not GPS_DATA_PATH.exists():
        print(f"ERROR: Data file not found: {GPS_DATA_PATH}")
        print("Please update GPS_DATA_PATH to point to your processed GPS parquet file.")
        return

    # Load and prepare data
    gps_data = load_and_filter_data(GPS_DATA_PATH, BBOX)
    print(f"Loaded {len(gps_data)} points within bounding box")

    if len(gps_data) == 0:
        print("ERROR: No data points within specified bounding box.")
        print("Please adjust BBOX coordinates to match your data extent.")
        return

    # Add simulated concentration if not present
    if "concentration" not in gps_data.columns:
        print("Adding simulated concentration values for demonstration...")
        gps_data = add_simulated_concentration(gps_data)

    # Create figure
    create_figure(gps_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()