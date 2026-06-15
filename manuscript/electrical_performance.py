"""
Victron Energy System Performance Figure
Figure dimensions: 7" x 4.5" (double column)

Produces a 2x2 panel figure characterising the CalMAPLab electrical system:
  (a) Power consumption (AC inverter + DC battery) vs ambient temperature
  (b) Internal vs ambient temperature with 1:1 reference line
  (c) Battery state of charge depletion during drives
  (d) Cumulative energy charged during shore-power sessions

Data sources:
  - Victron VRM CSV exports (three files through Feb 2026)
  - Ambient temperature from Airmar WX200 (bundled 5-min parquet, or L2 parquets)

Expected Victron CSV structure:
    Row 1: metadata (skipped), Row 2: column headers, Row 3: units (skipped),
    Row 4+: data.  Key columns include ac_consumption_l1/l2, current, voltage,
    output_voltage, output_current, grid_l1/l2, battery_soc, battery_power,
    ve_bus_state, temperature_1.

Dependencies:
    Required:  pandas, numpy, matplotlib, scipy, statsmodels, pyarrow (for --data-dir)

Usage:
    python manuscript/electrical_performance.py
    python manuscript/electrical_performance.py \\
        --victron-dir data/figure_inputs/electrical_performance/victron \\
        --amb-temp-file data/figure_inputs/electrical_performance/amb_temp_5min.parquet \\
        --output-dir figures/
"""

from __future__ import annotations

import argparse
import os
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_INPUTS = REPO_ROOT / "data/figure_inputs/electrical_performance"

# =============================================================================
# Configuration
# =============================================================================

# Colour palette
COLORS = {
    "inverter": "#404788",
    "battery": "#B63238",
}

# Ribbon style: "sd" = ± local residual SD, "se" = ± t·SE of the smooth
RIBBON_STAT = "sd"
RIBBON_MULTIPLIER = 1.0

# Victron CSV date column names (differ between export periods)
VICTRON_DATE_COLS = {
    "a": "america_los_angeles_07_00",
    "b": "america_los_angeles_08_00",
    "c": "america_los_angeles_07_00",
}

VICTRON_FILES = {
    "a": "UCBerkeley_log_20250501-0000_to_20251006-0820.csv",
    "b": "UCBerkeley_log_20251007-0000_to_20251105-2358.csv",
    "c": "UCBerkeley_log_20251106-0000_to_20260228-2359.csv",
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


def load_victron_data(victron_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all Victron CSV exports."""
    frames = []
    for key in ("a", "b", "c"):
        path = Path(victron_dir) / VICTRON_FILES[key]
        frames.append(load_victron_csv(str(path), VICTRON_DATE_COLS[key]))
    return pd.concat(frames, ignore_index=True)


def load_ambient_temp_from_l2(parquet_paths: list[str]) -> pd.DataFrame:
    """Load Airmar ambient temperature from L2a r2 parquets, aggregated to 5 min."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_paths, format="parquet")
    table = dataset.to_table(
        columns=["sample_time", "instrument", "parameter", "value"],
        filter=(pc.field("parameter") == "amb_T")
        & (pc.field("instrument") == "Airmar_WX200"),
    )
    df = table.to_pandas()
    df["sample_time"] = pd.to_datetime(df["sample_time"], utc=True)

    if df["sample_time"].dt.tz is not None:
        df["sample_time"] = (
            df["sample_time"]
            .dt.tz_convert("America/Los_Angeles")
            .dt.tz_localize(None)
        )

    return (
        df.assign(sample_time=lambda x: x["sample_time"].dt.floor("5min"))
        .groupby("sample_time", as_index=False)
        .agg(amb_temp_c=("value", "mean"))
    )


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


def _tricube(d):
    return np.clip((1.0 - np.abs(d) ** 3) ** 3, 0.0, 1.0)


def fit_lowess_with_bands(
    x, y, frac=0.3, ci=0.95, ribbon=None, ribbon_mult=None,
):
    """Local-linear LOWESS with shaded ribbons.

    ``ribbon="sd"`` (default): ``fit ± ribbon_mult * local residual SD`` —
    spread of measurements around the local trend.

    ``ribbon="se"``: ``fit ± ribbon_mult * t * SE(fit)`` — uncertainty in the
    smooth (R ``predict.loess(se=TRUE)``).
    """
    if ribbon is None:
        ribbon = RIBBON_STAT
    if ribbon_mult is None:
        ribbon_mult = RIBBON_MULTIPLIER

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 20:
        return np.array([]), np.array([]), None, None

    x_min, x_max = x.min(), x.max()
    xwidth = frac * (x_max - x_min)
    if xwidth <= 0:
        return np.array([]), np.array([]), None, None

    n_eval = min(200, max(50, len(x) // 10))
    x_smooth = np.linspace(x_min, x_max, n_eval)

    y_smooth = np.full(n_eval, np.nan)
    y_spread = np.full(n_eval, np.nan)
    dof_vals = np.zeros(n_eval)

    for i, xi in enumerate(x_smooth):
        w = _tricube(np.abs(x - xi) / xwidth)
        mask = w > 1e-8
        if mask.sum() < 10:
            continue

        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        W = w[mask]
        XtWX = X.T @ (W[:, None] * X)
        XtWy = X.T @ (W * y[mask])
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            continue

        y_smooth[i] = beta[0] + beta[1] * xi
        fitted = X @ beta
        resid = y[mask] - fitted
        dof = max(1.0, mask.sum() - 2)
        dof_vals[i] = dof

        if ribbon == "sd":
            y_spread[i] = np.sqrt(np.sum(W * resid ** 2) / np.sum(W))
        else:
            sigma2 = np.sum(W * resid ** 2) / dof
            row = np.array([1.0, xi])
            leverage = row @ np.linalg.inv(XtWX) @ row
            y_spread[i] = np.sqrt(max(0.0, sigma2 * leverage))

    ok = np.isfinite(y_smooth) & np.isfinite(y_spread)
    if ok.sum() < 3:
        return x_smooth, y_smooth, None, None

    x_smooth = x_smooth[ok]
    y_smooth = y_smooth[ok]
    y_spread = y_spread[ok]
    dof_vals = dof_vals[ok]

    if ribbon == "sd":
        margin = ribbon_mult * y_spread
    else:
        t_crit = np.where(
            dof_vals >= 30,
            stats.norm.ppf(0.5 + ci / 2),
            stats.t.ppf(0.5 + ci / 2, dof_vals),
        )
        margin = ribbon_mult * t_crit * y_spread

    y_lower = y_smooth - margin
    y_upper = y_smooth + margin

    return x_smooth, y_smooth, y_lower, y_upper


def _smooth_lowess(x_pts, y_pts, frac):
    """LOWESS smooth of sparse (x, y) percentile points."""
    idx = np.argsort(x_pts)
    x_pts, y_pts = x_pts[idx], y_pts[idx]
    return sm.nonparametric.lowess(
        y_pts, x_pts, frac=frac, it=0, return_sorted=True,
    )


def fit_percentile_bands(
    x, y, n_bins=22, lower_pct=10, upper_pct=90, smooth_frac=0.45,
):
    """Binned median with p10/p90 ribbons, each LOWESS-smoothed.

    1. Compute median and percentiles of y in equal-width x bins.
    2. Run a separate LOWESS smoother through each percentile curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 20:
        return np.array([]), np.array([]), None, None

    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, medians, lowers, uppers = [], [], [], []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        if mask.sum() < 8:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        medians.append(np.median(y[mask]))
        lowers.append(np.percentile(y[mask], lower_pct))
        uppers.append(np.percentile(y[mask], upper_pct))

    if len(centers) < 4:
        return np.array([]), np.array([]), None, None

    centers = np.array(centers)
    med_s = _smooth_lowess(centers, np.array(medians), smooth_frac)
    lo_s = _smooth_lowess(centers, np.array(lowers), smooth_frac)
    hi_s = _smooth_lowess(centers, np.array(uppers), smooth_frac)

    x_out = med_s[:, 0]
    y_out = med_s[:, 1]
    y_lower = np.interp(x_out, lo_s[:, 0], lo_s[:, 1])
    y_upper = np.interp(x_out, hi_s[:, 0], hi_s[:, 1])
    y_lo = np.minimum(y_lower, y_upper)
    y_hi = np.maximum(y_lower, y_upper)

    return x_out, y_out, y_lo, y_hi


def plot_lowess_with_band(ax, x, y, y_lower, y_upper, *, color, label=None, linewidth=1.5):
    """Plot a LOWESS line with a shaded ribbon."""
    ax.plot(x, y, color=color, linewidth=linewidth, label=label, zorder=3)
    if y_lower is None or y_upper is None:
        return
    ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2, linewidth=0, zorder=2)


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
    merged = merged.dropna(subset=["amb_temp_c"]).query("amb_temp_c >= 0 and amb_temp_c < 40")
    return merged


PANEL_A_POWER_MIN_W = 1000
PANEL_AB_TEMP_RANGE = (5, 35)
PERCENTILE_BAND_KWARGS = dict(n_bins=22, lower_pct=10, upper_pct=90, smooth_frac=0.45)


def _panel_ab_data(victron, amb_temp):
    """Shared driving subset for panels (a) and (b): same ambient temperature range."""
    lo, hi = PANEL_AB_TEMP_RANGE
    merged = _prepare_power_temp(victron, amb_temp)
    return merged.query(
        "power_ac > @PANEL_A_POWER_MIN_W and power_dc > @PANEL_A_POWER_MIN_W"
        " and amb_temp_c >= @lo and amb_temp_c <= @hi"
    ).dropna(subset=["internal_temp_c"])


def create_panel_a(ax, plot_data):
    """(a) Power consumption vs ambient temperature."""
    for col, color, label in [
        ("power_ac", COLORS["inverter"], "Inverter output"),
        ("power_dc", COLORS["battery"], "Battery output"),
    ]:
        x = plot_data["amb_temp_c"].values
        y = plot_data[col].values
        if len(x) > 20:
            xs, ys, yl, yu = fit_percentile_bands(x, y, **PERCENTILE_BAND_KWARGS)
            yl = np.clip(yl, 1500, None)
            plot_lowess_with_band(ax, xs, ys, yl, yu, color=color, label=label)

    ax.legend(loc="lower right", frameon=False, fontsize=7)
    lo, hi = PANEL_AB_TEMP_RANGE
    ax.set(xlim=(lo, hi), ylim=(1500, 4000),
           xlabel="Ambient temperature (°C)", ylabel="Power consumption (W)")
    ax.set_xticks(range(lo, hi + 1, 5))
    ax.set_yticks(range(1500, 4001, 500))
    add_panel_label(ax, "(a)")


def create_panel_b(ax, plot_data):
    """(b) Internal vs ambient temperature."""
    lo, hi = PANEL_AB_TEMP_RANGE
    ax.plot([lo, hi], [lo, hi], ls=":", color="#888888", lw=0.75, zorder=1, label="1:1 line")

    x = plot_data["amb_temp_c"].values
    y = plot_data["internal_temp_c"].values

    if len(x) > 20:
        xs, ys, yl, yu = fit_percentile_bands(x, y, **PERCENTILE_BAND_KWARGS)
        plot_lowess_with_band(
            ax, xs, ys, yl, yu, color=COLORS["inverter"], label="Internal temp.",
        )

    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set(xlim=(lo, hi), ylim=(lo, hi),
           xlabel="Ambient temperature (°C)", ylabel="Internal temperature (°C)")
    ax.set_xticks(range(lo, hi + 1, 5))
    ax.set_yticks(range(lo, hi + 1, 5))
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
        mask = (x >= 0) & (x <= 12) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        if len(x) > 20:
            xs, ys, yl, yu = fit_percentile_bands(
                x, y, n_bins=24, lower_pct=10, upper_pct=90, smooth_frac=0.45,
            )
            plot_lowess_with_band(ax, xs, ys, yl, yu, color=COLORS["inverter"])

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
            xs, ys, yl, yu = fit_percentile_bands(
                x, y, n_bins=18, lower_pct=10, upper_pct=90, smooth_frac=0.45,
            )
            plot_lowess_with_band(ax, xs, ys, yl, yu, color=COLORS["inverter"])

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

    ab_data = _panel_ab_data(victron, amb_temp)

    print("  Panel (a)...")
    create_panel_a(axes[0, 0], ab_data)
    print("  Panel (b)...")
    create_panel_b(axes[0, 1], ab_data)
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
        "--victron-dir",
        default=str(FIGURE_INPUTS / "victron"),
        help="Directory containing Victron CSV exports "
             "(default: data/figure_inputs/electrical_performance/victron)",
    )
    parser.add_argument(
        "--amb-temp-file",
        default=str(FIGURE_INPUTS / "amb_temp_5min.parquet"),
        help="Pre-built ambient temperature parquet "
             "(default: data/figure_inputs/electrical_performance/amb_temp_5min.parquet)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional: derive ambient temp from L2a r2 parquets in this directory "
             "(overrides --amb-temp-file)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "figures"),
        help="Output directory (default: figures/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    victron_dir = Path(args.victron_dir)
    for key in ("a", "b", "c"):
        path = victron_dir / VICTRON_FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"Victron {key.upper()} not found: {path}")

    print("Loading Victron data (3 CSV exports)...")
    victron = load_victron_data(victron_dir)
    print(f"  {len(victron):,} rows, {victron['date'].min()} – {victron['date'].max()}")

    compute_power_statistics(victron)

    if args.data_dir:
        parquet_paths = sorted(
            str(p) for p in Path(args.data_dir).glob("*_L2a_r2.parquet")
        )
        if not parquet_paths:
            raise FileNotFoundError(
                f"No L2a r2 parquet files found in {args.data_dir}"
            )
        print(f"Loading ambient temperature from {len(parquet_paths)} parquet files...")
        amb_temp = load_ambient_temp_from_l2(parquet_paths)
    else:
        amb_path = Path(args.amb_temp_file)
        if not amb_path.exists():
            raise FileNotFoundError(f"Ambient temperature file not found: {amb_path}")
        print(f"Loading ambient temperature from {amb_path}...")
        amb_temp = pd.read_parquet(amb_path)

    print(f"  {len(amb_temp):,} 5-min intervals")

    print("Building figure...")
    make_figure(victron, amb_temp, args.output_dir)


if __name__ == "__main__":
    main()