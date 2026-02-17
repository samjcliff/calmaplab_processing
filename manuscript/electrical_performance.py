"""
Victron Energy System Performance Figure
Figure dimensions: 7" x 4.5" (double column)

Produces a 2x2 panel figure characterising the CalMAPLab electrical system:
  (a) Power consumption (AC inverter + DC battery) vs ambient temperature
  (b) Internal vs ambient temperature with 1:1 reference line
  (c) Battery state of charge depletion during drives
  (d) Cumulative energy charged during shore-power sessions

Data sources:
  - Victron VRM CSV exports (two files covering the measurement period)
  - DuckDB database for co-located ambient temperature from the Airmar WX200

Expected Victron CSV structure:
    Row 1: metadata (skipped), Row 2: column headers, Row 3: units (skipped),
    Row 4+: data.  Key columns include ac_consumption_l1/l2, current, voltage,
    output_voltage, output_current, grid_l1/l2, battery_soc, battery_power,
    ve_bus_state, temperature_1.

Expected database schema (table: measurements):
    sample_time (TIMESTAMPTZ), parameter (VARCHAR), value (DOUBLE),
    instrument (VARCHAR).  Ambient temperature uses parameter='amb_T',
    instrument='Airmar_WX200'.

Dependencies:
    Required:  duckdb, pandas, numpy, matplotlib, scipy, statsmodels

Usage:
    python electrical_performance.py \\
        --db data/drives_r2.duckdb \\
        --victron-dir data/raw/victron/ \\
        --output-dir figures/
"""

import argparse
import os
from io import StringIO
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# =============================================================================
# Configuration
# =============================================================================

# Colour palette
COLORS = {
    "inverter": "#404788",
    "battery": "#B63238",
}

# Victron CSV date column names (differ between export periods)
VICTRON_DATE_COLS = {
    "a": "america_los_angeles_07_00",
    "b": "america_los_angeles_08_00",
}

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

def load_victron_csv(filepath, date_col_name):
    """Load a Victron VRM CSV export with its non-standard header layout.

    The CSV has: row 1 = metadata, row 2 = headers, row 3 = units, row 4+ = data.
    We keep rows 2 and 4+ and normalise column names.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Keep header (line index 1) and data (line index 3+)
    lines_clean = [lines[1]] + lines[3:]

    df = pd.read_csv(StringIO("".join(lines_clean)))

    df.columns = (
        df.columns.str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df = df.rename(columns={date_col_name: "date"})
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    return df


def load_victron_data(path_a, path_b):
    """Load and concatenate both Victron CSV files."""
    victron_a = load_victron_csv(path_a, VICTRON_DATE_COLS["a"])
    victron_b = load_victron_csv(path_b, VICTRON_DATE_COLS["b"])
    return pd.concat([victron_a, victron_b], ignore_index=True)


def load_ambient_temp(con):
    """Query ambient temperature from the database and agg to 5 mins."""
    query = """
        SELECT sample_time, parameter, value, instrument
        FROM measurements
        WHERE parameter = 'amb_T'
          AND instrument = 'Airmar_WX200'
    """
    df = con.execute(query).fetchdf()

    if df["sample_time"].dt.tz is not None:
        df["sample_time"] = (
            df["sample_time"]
            .dt.tz_convert("America/Los_Angeles")
            .dt.tz_localize(None)
        )

    amb_temp = (
        df.assign(sample_time=lambda x: x["sample_time"].dt.floor("5min"))
        .groupby("sample_time", as_index=False)
        .agg(amb_temp_c=("value", "mean"))
    )
    return amb_temp


# =============================================================================
# Statistical Helpers
# =============================================================================

def add_panel_label(ax, label):
    """Place a bold panel label outside the top-left corner of an axes."""
    ax.text(
        -0.18, 1.12, label,
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="top", ha="left",
    )


def fit_lowess_with_bands(x, y, frac=0.3, n_bins=25, lower_pct=10, upper_pct=90):
    """Fit a LOWESS smoother with quantile-based confidence bands.

    The main curve is LOWESS on all data.  Bands are computed by binning the
    data along x, taking the lower_pct and upper_pct percentiles in each bin,
    then smoothing those percentile series with LOWESS.

    Returns
    -------
    x_smooth, y_smooth, y_lower, y_upper : arrays (y_lower/y_upper may be None)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]

    lowess_result = sm.nonparametric.lowess(y_sorted, x_sorted, frac=frac)
    x_smooth = lowess_result[:, 0]
    y_smooth = lowess_result[:, 1]

    # Binned percentiles for bands
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers, lower_vals, upper_vals = [], [], []

    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if mask.sum() > 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            lower_vals.append(np.percentile(y[mask], lower_pct))
            upper_vals.append(np.percentile(y[mask], upper_pct))

    if len(bin_centers) < 3:
        return x_smooth, y_smooth, None, None

    bin_centers = np.array(bin_centers)
    lower_smooth = sm.nonparametric.lowess(np.array(lower_vals), bin_centers, frac=frac)
    upper_smooth = sm.nonparametric.lowess(np.array(upper_vals), bin_centers, frac=frac)

    y_lower = np.interp(x_smooth, lower_smooth[:, 0], lower_smooth[:, 1])
    y_upper = np.interp(x_smooth, upper_smooth[:, 0], upper_smooth[:, 1])

    return x_smooth, y_smooth, y_lower, y_upper


# =============================================================================
# Power Statistics (printed summary)
# =============================================================================

def compute_power_statistics(victron):
    """Compute and print summary power statistics across all drives."""
    data = victron.copy()
    data["power_ac"] = data["ac_consumption_l2"] + data["ac_consumption_l1"]
    data["power_dc"] = -data["current"] * data["voltage"]
    data["alternator_power"] = data["output_voltage"] * data["output_current"]
    data["shore_power"] = data["grid_l1"] + data["grid_l2"]
    data["day"] = data["date"].dt.floor("D").astype(str)

    driving = data.query('ve_bus_state == "Inverting" and power_ac > 1300').copy()

    print("\n" + "=" * 60)
    print("POWER STATISTICS")
    print("=" * 60)

    if len(driving) > 0:
        daily = (
            driving.groupby("day", as_index=False)
            .agg(
                mean_power_ac=("power_ac", "mean"),
                mean_power_dc=("power_dc", "mean"),
                max_power_ac=("power_ac", "max"),
                max_power_dc=("power_dc", "max"),
                mean_alternator=("alternator_power", "mean"),
            )
        )
        print("\nInverter Output (AC) - Daily Means:")
        print(f"  Low:  {daily['mean_power_ac'].min():.0f} W")
        print(f"  High: {daily['mean_power_ac'].max():.0f} W")
        print(f"  Overall: {daily['mean_power_ac'].mean():.0f} W")

        print("\nBattery Output (DC) - Daily Means:")
        print(f"  Low:  {daily['mean_power_dc'].min():.0f} W")
        print(f"  High: {daily['mean_power_dc'].max():.0f} W")
        print(f"  Overall: {daily['mean_power_dc'].mean():.0f} W")

        print("\nPeak Power (across all drives):")
        print(f"  Inverter: {driving['power_ac'].max():.0f} W")
        print(f"  Battery:  {driving['power_dc'].max():.0f} W")

        print("\nAlternator Draw (during drives):")
        print(f"  Average: {daily['mean_alternator'].mean():.0f} W")
        print(f"  Range: {daily['mean_alternator'].min():.0f} – {daily['mean_alternator'].max():.0f} W")

    charging = data.query('ve_bus_state in ["Bulk", "Absorption", "Float"]')
    if len(charging) > 0:
        print("\nShore Power (during charging):")
        print(f"  Max:  {charging['shore_power'].max():.0f} W")
        print(f"  Mean: {charging['shore_power'].mean():.0f} W")

    print("=" * 60 + "\n")


# =============================================================================
# Panel Functions
# =============================================================================

def _prepare_power_temp(victron, amb_temp):
    """Shared preprocessing for panels (a) and (b): merge power data with ambient 
    temperature on 5-min intervals during driving (AC > 1300 W for valid drives)."""
    df = victron.copy()
    df["power_ac"] = df["ac_consumption_l2"] + df["ac_consumption_l1"]
    df["power_dc"] = -df["current"] * df["voltage"]

    df = df.dropna(subset=["power_ac"]).query("power_ac > 1300")
    df["date"] = df["date"].dt.floor("5min")

    agg = (
        df.groupby("date", as_index=False)
        .agg(
            power_ac=("power_ac", "mean"),
            power_dc=("power_dc", "mean"),
            internal_temp_c=("temperature_1", "mean"),
        )
        .rename(columns={"date": "sample_time"})
    )

    merged = agg.merge(amb_temp, on="sample_time", how="left")
    merged = merged.dropna(subset=["amb_temp_c"]).query("amb_temp_c > 10 and amb_temp_c < 40")
    return merged


def create_panel_a(ax, victron, amb_temp):
    """(a) Power consumption vs ambient temperature."""
    merged = _prepare_power_temp(victron, amb_temp)
    plot_data = merged.query("power_dc > 1500")

    for col, color, label in [
        ("power_ac", COLORS["inverter"], "Inverter output"),
        ("power_dc", COLORS["battery"], "Battery output"),
    ]:
        mask = plot_data[["amb_temp_c", col]].notna().all(axis=1)
        x, y = plot_data.loc[mask, "amb_temp_c"].values, plot_data.loc[mask, col].values
        if len(x) > 20:
            xs, ys, yl, yu = fit_lowess_with_bands(x, y, frac=0.6)
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
            if yl is not None:
                ax.fill_between(xs, yl, yu, color=color, alpha=0.2)

    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set(xlim=(10, 35), ylim=(1500, 4000),
           xlabel="Ambient temperature (°C)", ylabel="Power consumption (W)")
    ax.set_xticks(range(10, 36, 5))
    ax.set_yticks(range(1500, 4001, 500))
    add_panel_label(ax, "(a)")


def create_panel_b(ax, victron, amb_temp):
    """(b) Internal vs ambient temperature."""
    merged = _prepare_power_temp(victron, amb_temp)

    ax.plot([10, 35], [10, 35], ls=":", color="#888888", lw=0.75, zorder=1, label="1:1 line")

    mask = merged[["amb_temp_c", "internal_temp_c"]].notna().all(axis=1)
    x = merged.loc[mask, "amb_temp_c"].values
    y = merged.loc[mask, "internal_temp_c"].values

    if len(x) > 20:
        xs, ys, yl, yu = fit_lowess_with_bands(x, y, frac=0.4)
        ax.plot(xs, ys, color=COLORS["inverter"], linewidth=1.5, label="Internal temp.")
        if yl is not None:
            ax.fill_between(xs, yl, yu, color=COLORS["inverter"], alpha=0.2)

    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set(xlim=(10, 35), ylim=(10, 35),
           xlabel="Ambient temperature (°C)", ylabel="Internal temperature (°C)")
    ax.set_xticks(range(10, 36, 5))
    ax.set_yticks(range(10, 36, 5))
    add_panel_label(ax, "(b)")


def create_panel_c(ax, victron):
    """(c) Battery state of charge depletion during drives."""
    df = victron.query('battery_soc < 100 and ve_bus_state == "Inverting"').copy()
    df["day"] = df["date"].dt.floor("D").astype(str)
    df["time_s"] = (
        df["date"].dt.hour * 3600 + df["date"].dt.minute * 60 + df["date"].dt.second
    )

    # Keep only drives with monotonically decreasing SoC
    valid_days = []
    for _, group in df.groupby("day"):
        group = group.sort_values("time_s").copy()
        group["time_s"] = group["time_s"] - group["time_s"].min()
        if not (group["battery_soc"].diff() > 0).any():
            valid_days.append(group)

    if valid_days:
        combined = pd.concat(valid_days, ignore_index=True)
        x = (combined["time_s"] / 3600).values
        y = combined["battery_soc"].values
        mask = (x >= 0) & (x <= 14) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        if len(x) > 20:
            xs, ys, yl, yu = fit_lowess_with_bands(x, y, frac=0.4)
            ax.plot(xs, ys, color=COLORS["inverter"], linewidth=1.5)
            if yl is not None:
                ax.fill_between(xs, yl, yu, color=COLORS["inverter"], alpha=0.2)

    ax.set(xlim=(0, 12), ylim=(0, 100),
           xlabel="Hours into drive", ylabel="Battery state of charge (%)")
    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks(range(0, 101, 20))
    add_panel_label(ax, "(c)")


def create_panel_d(ax, victron):
    """(d) Cumulative energy charged during shore-power sessions."""
    df = victron.query(
        'battery_soc < 100 and ve_bus_state in ["Bulk", "Absorption"]'
    ).copy()
    df["day"] = df["date"].dt.floor("D").astype(str)
    df["time_s"] = (
        df["date"].dt.hour * 3600 + df["date"].dt.minute * 60 + df["date"].dt.second
    )

    valid_days = []
    charging_rates = []

    for _, group in df.groupby("day"):
        group = group.sort_values("time_s").copy()
        group["time_s"] = group["time_s"] - group["time_s"].min()

        # Skip days with large time gaps (>1000 s)
        if (group["time_s"].diff() > 1000).any():
            continue

        group["charge_power_kwh"] = group["battery_power"] / 12000
        group["cumulative_charge_kwh"] = (
            group["charge_power_kwh"].cumsum() - group["charge_power_kwh"].cumsum().min()
        )
        valid_days.append(group)

        total = group["cumulative_charge_kwh"].max()
        hours = (group["time_s"].max() - group["time_s"].min()) / 3600
        if hours > 0:
            charging_rates.append(total / hours)

    if charging_rates:
        print(f"Charging rate: median {np.median(charging_rates):.2f}, "
              f"SD {np.std(charging_rates):.2f} kWh/hour")

    if valid_days:
        combined = pd.concat(valid_days, ignore_index=True)
        x = (combined["time_s"] / 3600).values
        y = combined["cumulative_charge_kwh"].values
        mask = (x >= 0) & (x <= 6) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        if len(x) > 20:
            xs, ys, yl, yu = fit_lowess_with_bands(x, y, frac=0.6)
            ax.plot(xs, ys, color=COLORS["inverter"], linewidth=1.5)
            if yl is not None:
                ax.fill_between(xs, yl, yu, color=COLORS["inverter"], alpha=0.2)

    ax.set(xlim=(0, 6), ylim=(0, None),
           xlabel="Hours charging", ylabel="Cumulative charge (kWh)")
    ax.set_xticks(range(0, 7, 1))
    add_panel_label(ax, "(d)")


# =============================================================================
# Main Figure
# =============================================================================

def make_figure(victron, amb_temp, output_dir):
    """Build the 2x2 panel figure and save to disk."""
    setup_amt_style()

    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))
    fig.subplots_adjust(
        left=0.10, right=0.97, top=0.92, bottom=0.12,
        wspace=0.30, hspace=0.42,
    )

    print("  Panel (a)...")
    create_panel_a(axes[0, 0], victron, amb_temp)
    print("  Panel (b)...")
    create_panel_b(axes[0, 1], victron, amb_temp)
    print("  Panel (c)...")
    create_panel_c(axes[1, 0], victron)
    print("  Panel (d)...")
    create_panel_d(axes[1, 1], victron)

    os.makedirs(output_dir, exist_ok=True)

    for fmt in ("pdf", "png"):
        path = os.path.join(output_dir, f"electrical_performance.{fmt}")
        print(f"Saving {fmt.upper()}: {path}")
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print("Done!")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the Victron electrical system performance figure (AMT style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default="data/drives_r2.duckdb",
        help="Path to DuckDB database (default: data/drives_r2.duckdb)",
    )
    parser.add_argument(
        "--victron-dir",
        default="data/raw/victron",
        help="Directory containing Victron CSV exports (default: data/victron)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/",
        help="Output directory (default: figures/)",
    )
    return parser.parse_args()


# Known Victron CSV filenames
VICTRON_FILES = {
    "a": "UCBerkeley_log_20250501-0000_to_20251006-0820.csv",
    "b": "UCBerkeley_log_20251007-0000_to_20251105-2358.csv",
}


def main():
    args = parse_args()

    path_a = os.path.join(args.victron_dir, VICTRON_FILES["a"])
    path_b = os.path.join(args.victron_dir, VICTRON_FILES["b"])

    for label, path in [("Database", args.db), ("Victron A", path_a),
                         ("Victron B", path_b)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    print("Loading Victron data...")
    victron = load_victron_data(path_a, path_b)

    compute_power_statistics(victron)

    print("Loading ambient temperature from database...")
    con = duckdb.connect(args.db, read_only=True)
    amb_temp = load_ambient_temp(con)
    con.close()

    print("Building figure...")
    make_figure(victron, amb_temp, args.output_dir)


if __name__ == "__main__":
    main()