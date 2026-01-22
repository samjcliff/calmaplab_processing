"""
VOCUS Calibration Processing Pipeline
======================================

Processing code for VOCUS PTR-ToF calibrations.

This module handles:
1. Loading processed CPS data
2. Identifying calibration windows (multi-step dilution calibrations)
3. Computing Deming regression for each species
4. Estimating k_PTR sensitivity relationships
5. Processing in-drive zero measurements
6. Generating diagnostic plots

Typical integration with instrument_pipeline.py.

Author: S. J. Cliff
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Optional imports for plotting and regression
try:
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf_backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available - diagnostic plots disabled")

try:
    from scipy import stats as scipy_stats
    from scipy.odr import ODR, Model, RealData
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - Deming regression will use OLS fallback")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VOCUSCalConfig:
    """Configuration parameters for VOCUS calibration processing."""
    
    # Default k_PTR value for sensitivity estimation (units: 10^-9 cm³ molec⁻¹ s⁻¹)
    default_k: float = 2.5
    
    # Calibration step flow rates to check (sccm)
    cal_step_flows: List[float] = field(default_factory=lambda: [0, 2, 5, 10])
    
    # Flow tolerance for step identification (sccm)
    flow_tolerance: float = 0.1
    
    # Minimum zero flow threshold (sccm)
    zero_flow_thresh: float = 450.0
    
    # Minimum samples required for a valid calibration window
    min_samples: int = 70
    
    # Samples to use from end of window (excluding last 10)
    window_samples: int = 60
    
    # Gap threshold for grouping zero periods (seconds)
    zero_gap_threshold: float = 30.0
    
    # Species columns range (start, end) - 0-indexed
    # In R this was columns 2:661 (1-indexed), so 1:661 in 0-indexed
    species_col_range: Tuple[int, int] = (1, 661)
    
    # Calibration cylinder number
    cal_cylinder_no: str = ""


@dataclass
class VOCUSCalPathConfig:
    """File paths for VOCUS calibration processing."""
    
    # Input path for processed CPS files
    input_path: str = ""
    
    # Output path for calibrated files
    output_path: str = ""
    
    # Path for diagnostic plots
    diagnostic_plots_path: str = ""
    
    # Path for calibration statistics
    cal_stats_path: str = ""
    
    # Path to calibration standards CSV (cylinder concentrations)
    cal_standard_path: str = ""
    
    # Path to k_PTR sensitivity standards CSV
    ksens_standards_path: str = ""


# =============================================================================
# k_PTR Sensitivity Data
# =============================================================================

def load_ksens_standards(filepath: str, cal_cylinder_no: str) -> pd.DataFrame:
    """
    Load k_PTR sensitivity data from CSV file.
    
    Expected CSV columns: cylinder_number, species, names, ksens, mq
    
    Parameters
    ----------
    filepath : str
        Path to ksens standards CSV file
    cal_cylinder_no : str
        Calibration cylinder identifier to filter by
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: species, names, ksens, mq
    """
    print(f"  Loading k_PTR standards: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Filter to requested cylinder
    df_filtered = df[df['cylinder_number'] == cal_cylinder_no].copy()
    
    if len(df_filtered) == 0:
        warnings.warn(f"No k_PTR data found for cylinder {cal_cylinder_no}")
        return pd.DataFrame(columns=['species', 'names', 'ksens', 'mq'])
    
    # Return only the columns we need
    result = df_filtered[['species', 'names', 'ksens', 'mq']].copy()
    
    print(f"    Loaded {len(result)} species for cylinder {cal_cylinder_no}")
    
    return result


def get_ksens_data(cal_cylinder_no: str, ksens_path: Optional[str] = None) -> pd.DataFrame:
    """
    Get k_PTR sensitivity data for a calibration cylinder.
    
    If ksens_path is provided, loads from CSV file.
    Otherwise falls back to hardcoded defaults (deprecated).
    
    Parameters
    ----------
    cal_cylinder_no : str
        Calibration cylinder identifier
    ksens_path : str, optional
        Path to ksens standards CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: species, names, ksens, mq
    """
    # If path provided, load from CSV
    if ksens_path is not None and Path(ksens_path).exists():
        return load_ksens_standards(ksens_path, cal_cylinder_no)
    
    # Fallback to hardcoded values (deprecated - for backwards compatibility)
    warnings.warn(
        "Using hardcoded k_PTR values. Consider providing ksens_standards_path "
        "in VOCUSCalPathConfig for better maintainability.",
        DeprecationWarning
    )
    
    if cal_cylinder_no == "CC524064":
        data = {
            'species': [
                "C2H5O", "CH5O", "CH5S", "C2H7O", "C2H4N", "C3H7O", "C3H4N", "C5H9",
                "C2H7S", "C4H7O", "C4H9O", "C6H7", "C7H9", "C6H19O3Si3", "C8H11",
                "C10H17", "C9H13", "C10H23", "C10H31O5Si5"
            ],
            'names': [
                "Acetaldehyde", "Methanol", "Methanethiol", "Ethanol", "Acetonitrile",
                "Acetone", "Acrylonitrile", "Isoprene", "DMS", "MVK", "MEK", "Benzene",
                "Toluene", "D3", "Xylene", "Monoterpenes", "TMB", "Decane", "D5"
            ],
            'ksens': [
                2.97e-9, 2.31e-9, 2.21e-9, 2.37e-9, 4.03e-9, 3.24e-9, 4.17e-9, 1.93e-9,
                2.26e-9, 3.51e-9, 3.23e-9, 1.93e-9, 2.07e-9, 3.20e-9, 2.21e-9, 2.48e-9,
                2.42e-9, 2.5e-9, 3.5e-9
            ],
            'mq': [
                45, 33, 49, 47, 42, 59, 54, 69, 63, 71, 73, 79, 93, 223, 107, 137, 121, 143, 371
            ]
        }
    elif cal_cylinder_no == "CC515288":
        data = {
            'species': [
                "C2H5O", "CH5O", "C2H7O", "C3H7O", "C3H4N", "C5H9", "C4H7O", "C4H9O",
                "C7H9", "C8H11", "C10H17", "C9H13", "C10H23", "C10H31O5Si5", "C6H15"
            ],
            'names': [
                "Acetaldehyde", "Methanol", "Ethanol", "Acetone", "Acrylonitrile",
                "Isoprene", "MVK", "MEK", "Toluene", "Xylene", "Monoterpenes", "TMB",
                "Decane", "D5", "Triethylamine"
            ],
            'ksens': [
                2.97e-9, 2.31e-9, 2.37e-9, 3.24e-9, 4.17e-9, 1.93e-9, 3.51e-9, 3.23e-9,
                2.07e-9, 2.21e-9, 2.48e-9, 2.42e-9, 2.5e-9, 3.5e-9, 3.0e-9
            ],
            'mq': [
                45, 33, 47, 59, 54, 69, 71, 73, 93, 107, 137, 121, 143, 371, 102
            ]
        }
    else:
        warnings.warn(f"Unknown cylinder {cal_cylinder_no}, returning empty ksens data")
        data = {'species': [], 'names': [], 'ksens': [], 'mq': []}
    
    return pd.DataFrame(data)


# =============================================================================
# Part 1: Data Loading
# =============================================================================

def load_cps_data(filepath: str) -> pd.DataFrame:
    """
    Load processed VOCUS CPS data.
    
    Parameters
    ----------
    filepath : str
        Path to CPS CSV file
    
    Returns
    -------
    pd.DataFrame
        CPS data with parsed datetime
    """
    print(f"  Loading CPS data: {filepath}")
    
    df = pd.read_parquet(filepath)
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
    
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def load_calibration_standards(filepath: str) -> pd.DataFrame:
    """
    Load calibration standards from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to calibration standards CSV
    
    Returns
    -------
    pd.DataFrame
        Long-format calibration standards
    """
    print(f"  Loading calibration standards: {filepath}")
    
    cs = pd.read_csv(filepath)
    
    # Get ID columns (first 2) and measurement columns (rest)
    id_cols = list(cs.columns[:2])
    measure_cols = list(cs.columns[2:])
    
    # Melt to long format
    cs_long = cs.melt(
        id_vars=id_cols,
        value_vars=measure_cols,
        var_name='species_full',
        value_name='cal_concs'
    )
    
    # Extract species name (remove last 4 chars, e.g., "_ppb")
    cs_long['species'] = cs_long['species_full'].str[:-4]
    
    print(f"    Loaded standards for {cs_long['species'].nunique()} species")
    
    return cs_long


# =============================================================================
# Part 2: Calibration Window Detection
# =============================================================================

def find_cal_window(
    df: pd.DataFrame,
    step_flow: float,
    cal_valve_status: int,
    zero_valve_status: int,
    config: VOCUSCalConfig,
    cal_cyl_df: pd.DataFrame,
    species_cols: List[str]
) -> Optional[pd.DataFrame]:
    """
    Find and extract mean values for a calibration step window.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format CPS data
    step_flow : float
        Target calibration flow rate
    cal_valve_status : int
        Expected cal_valve state (0 or 1)
    zero_valve_status : int
        Expected zero_valve state (0 or 1)
    config : VOCUSCalConfig
        Configuration parameters
    cal_cyl_df : pd.DataFrame
        Calibration cylinder concentrations
    species_cols : list
        List of species column names
    
    Returns
    -------
    pd.DataFrame or None
        Mean values for the calibration window, or None if insufficient data
    """
    # Find matching rows
    mask = (
        (np.abs(df['cal_flow'] - step_flow) < config.flow_tolerance) &
        (df['cal_valve'] == cal_valve_status) &
        (df['zero_valve'] == zero_valve_status) &
        (df['zero_flow'] > config.zero_flow_thresh)
    )
    
    idx = df.index[mask].tolist()
    
    if len(idx) < config.min_samples:
        return None
    
    # Keep last min_samples, then take first window_samples (skip last 10s)
    idx = idx[-config.min_samples:]
    idx = idx[:config.window_samples]
    
    # Subset data
    sub = df.loc[idx, ['date', 'cal_flow', 'zero_flow'] + species_cols].copy()
    
    # Melt to long format
    sub_long = sub.melt(
        id_vars=['date', 'cal_flow', 'zero_flow'],
        value_vars=species_cols,
        var_name='species',
        value_name='counts'
    )
    
    # Join with calibrant concentrations
    sub_long = sub_long.merge(cal_cyl_df[['species', 'cal_concs']], on='species', how='inner')
    
    if len(sub_long) == 0:
        return None
    
    # Compute diluted concentration
    sub_long['cal_conc_dil'] = sub_long['cal_concs'] * (sub_long['cal_flow'] / sub_long['zero_flow'])
    
    # Aggregate means by species
    result = sub_long.groupby('species').agg({
        'counts': 'mean',
        'cal_flow': 'mean',
        'zero_flow': 'mean',
        'cal_conc_dil': 'mean',
        'date': lambda x: pd.Timestamp(x.astype('int64').mean(), unit='ns', tz='UTC')
    }).reset_index()
    
    return result


# =============================================================================
# Part 3: Deming Regression
# =============================================================================

def deming_regression(x: np.ndarray, y: np.ndarray, ratio: float = 1.0) -> Tuple[float, float]:
    """
    Perform Deming regression (orthogonal distance regression).
    
    Parameters
    ----------
    x : array
        Independent variable
    y : array
        Dependent variable
    ratio : float
        Ratio of error variances (var_y / var_x)
    
    Returns
    -------
    tuple
        (intercept, slope)
    """
    if not HAS_SCIPY:
        # Fallback to OLS
        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
        return intercept, slope
    
    # Remove NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return np.nan, np.nan
    
    # Deming regression using orthogonal distance regression
    def linear_func(B, x):
        return B[0] + B[1] * x
    
    model = Model(linear_func)
    
    # Estimate initial parameters using OLS
    slope_init, intercept_init, _, _, _ = scipy_stats.linregress(x, y)
    
    # Set up data with error estimates
    # For Deming, we assume equal relative errors
    sx = np.std(x) * 0.1 if np.std(x) > 0 else 1.0
    sy = np.std(y) * 0.1 if np.std(y) > 0 else 1.0
    
    data = RealData(x, y, sx=sx, sy=sy)
    
    odr = ODR(data, model, beta0=[intercept_init, slope_init])
    output = odr.run()
    
    intercept, slope = output.beta
    
    return intercept, slope


def compute_cal_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Deming regression statistics for each species.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Combined calibration window data
    
    Returns
    -------
    pd.DataFrame
        Regression results with intercept and slope per species
    """
    print("  Computing calibration statistics (Deming regression)...")
    
    cal_datetime = stats_df['date'].max()
    
    results = []
    for species, group in stats_df.groupby('species'):
        x = group['cal_conc_dil'].values
        y = group['counts'].values
        
        intercept, slope = deming_regression(x, y)
        
        results.append({
            'species': species,
            'intercept': intercept,
            'slope': slope,
            'date': cal_datetime
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"    Computed stats for {len(results_df)} species")
    
    return results_df


# =============================================================================
# Part 4: k_PTR Sensitivity Analysis
# =============================================================================

def compute_ksens_relationship(
    results_df: pd.DataFrame,
    ksens_df: pd.DataFrame,
    default_k: float,
    exclude_species: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Compute k_PTR vs sensitivity relationship.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Calibration regression results
    ksens_df : pd.DataFrame
        Known k_PTR values
    default_k : float
        Default k_PTR for unknown species
    exclude_species : list, optional
        Species to exclude from regression (poor BSQ transmission)
    
    Returns
    -------
    tuple
        (merged_df, ksens_value, plot_df)
    """
    if exclude_species is None:
        exclude_species = ["CH5S", "CH5O", "C2H7O", "C10H23"]
    
    print("  Computing k_PTR sensitivity relationship...")
    
    # Merge with known ksens values
    merged = ksens_df.merge(results_df, on='species', how='inner')
    
    if len(merged) == 0:
        print("    WARNING: No matching species for ksens analysis")
        return merged, default_k * 1e9, merged
    
    # Separate excluded species
    plot_df = merged[~merged['species'].isin(exclude_species)].copy()
    
    if len(plot_df) < 2:
        print("    WARNING: Insufficient species for regression")
        ksens_value = default_k * 1e9  # Default fallback
        return merged, ksens_value, plot_df
    
    # Linear regression: slope ~ ksens
    x = plot_df['ksens'].values * 1e9  # Convert to 10^-9 units
    y = plot_df['slope'].values
    
    slope_lr, intercept_lr, _, _, _ = scipy_stats.linregress(x, y)
    
    # Estimate ksens_value for default_k
    ksens_value = default_k * slope_lr + intercept_lr
    
    print(f"    k_PTR relationship: slope={slope_lr:.2e}, intercept={intercept_lr:.2e}")
    print(f"    Estimated sensitivity for k={default_k}: {ksens_value:.2e}")
    
    return merged, ksens_value, plot_df


# =============================================================================
# Part 5: In-Drive Zero Processing
# =============================================================================

def process_in_drive_zeros(
    df: pd.DataFrame,
    config: VOCUSCalConfig,
    species_cols: List[str]
) -> pd.DataFrame:
    """
    Process in-drive zero measurements.
    
    Parameters
    ----------
    df : pd.DataFrame
        CPS data
    config : VOCUSCalConfig
        Configuration parameters
    species_cols : list
        Species column names
    
    Returns
    -------
    pd.DataFrame
        Aggregated zero measurements
    """
    print("  Processing in-drive zeros...")
    
    # Filter for zero conditions
    zero_mask = (
        (df['zero_valve'] == 1) &
        (df['cal_valve'] == 0) &
        (df['zero_flow'] > 400)
    )
    
    zero_df = df[zero_mask].copy().sort_values('date')
    
    if len(zero_df) == 0:
        print("    No zero periods found")
        return pd.DataFrame()
    
    # Group by time gaps (new group when gap > threshold)
    zero_df['date_numeric'] = zero_df['date'].astype('int64') / 1e9  # seconds
    zero_df['gap'] = zero_df['date_numeric'].diff()
    zero_df['flag'] = (zero_df['gap'].isna() | (zero_df['gap'] > config.zero_gap_threshold)).cumsum()
    
    # Aggregate each group
    results = []
    
    for flag, group in zero_df.groupby('flag'):
        if len(group) < config.min_samples:
            continue
        
        # Last 70 samples, drop last 10 -> use middle 60
        window = group.iloc[-(config.min_samples):-(config.min_samples - config.window_samples)]
        
        if len(window) == 0:
            continue
        
        # Compute means for numeric columns
        numeric_cols = [c for c in window.columns if window[c].dtype in ['float64', 'int64'] and c not in ['flag', 'gap', 'date_numeric']]
        
        row = {col: window[col].mean() for col in numeric_cols}
        row['date'] = pd.Timestamp(window['date'].astype('int64').mean(), unit='ns', tz='UTC')
        row['flag'] = flag
        
        results.append(row)
    
    if results:
        result_df = pd.DataFrame(results)
        print(f"    Found {len(result_df)} zero periods")
        return result_df
    else:
        print("    No valid zero periods found")
        return pd.DataFrame()


# =============================================================================
# Part 6: Diagnostic Plots
# =============================================================================

def plot_calibration_curves(
    stats_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate calibration curve diagnostic plots.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Calibration statistics
    output_path : str
        Path for output PDF
    """
    if not HAS_MATPLOTLIB:
        print("    Skipping plots (matplotlib not available)")
        return
    
    print("  Generating calibration curve plots...")
    
    species_list = stats_df['species'].unique()
    n_species = len(species_list)
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = int(np.ceil(n_species / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_species > 1 else [axes]
    
    for i, species in enumerate(species_list):
        ax = axes[i]
        data = stats_df[stats_df['species'] == species]
        
        x = data['cal_conc_dil'].values
        y = data['counts'].values
        
        # Plot points
        ax.scatter(x, y, s=30, alpha=0.7)
        
        # Fit and plot line
        if len(x) >= 2:
            slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
            x_line = np.array([x.min(), x.max()])
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, 'b-', alpha=0.7)
        
        ax.set_title(species, fontsize=10)
        ax.set_xlabel('Conc (ppb)', fontsize=8)
        ax.set_ylabel('Counts', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide empty subplots
    for i in range(n_species, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=150)
    plt.close()
    
    print(f"    Saved: {output_path}")


def plot_ksens_relationship(
    plot_df: pd.DataFrame,
    exclude_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate k_PTR vs sensitivity diagnostic plot.
    
    Parameters
    ----------
    plot_df : pd.DataFrame
        Data for regression (included species)
    exclude_df : pd.DataFrame
        Excluded species (poor BSQ transmission)
    output_path : str
        Path for output PDF
    """
    if not HAS_MATPLOTLIB:
        print("    Skipping plots (matplotlib not available)")
        return
    
    print("  Generating k_PTR sensitivity plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot included points
    x = plot_df['ksens'].values * 1e9
    y = plot_df['slope'].values
    ax.scatter(x, y, s=50, alpha=0.7, label='Included')
    
    # Fit and plot line
    if len(x) >= 2:
        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
        x_line = np.array([x.min(), x.max()])
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, 'b-', alpha=0.7)
    
    # Plot excluded points
    if len(exclude_df) > 0:
        x_ex = exclude_df['ksens'].values * 1e9
        y_ex = exclude_df['slope'].values
        ax.scatter(x_ex, y_ex, s=50, c='red', alpha=0.7, label='Excluded')
    
    ax.set_xlabel(r'$k_{PTR}$ / $10^{-9}$ cm$^3$ molec$^{-1}$ s$^{-1}$', fontsize=12)
    ax.set_ylabel(r'Sensitivity / cts ppb$^{-1}$', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=150)
    plt.close()
    
    print(f"    Saved: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def process_vocus_calibrations(
    date_str: str,
    config: VOCUSCalConfig,
    paths: VOCUSCalPathConfig
) -> Dict[str, Any]:
    """
    Process VOCUS calibrations for a single date.
    
    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format
    config : VOCUSCalConfig
        Calibration configuration
    paths : VOCUSCalPathConfig
        Input/output paths
    
    Returns
    -------
    dict
        Processing outputs
    """
    print(f"\n{'='*60}")
    print(f"Processing VOCUS calibrations for {date_str}")
    print(f"{'='*60}")
    
    outputs = {}
    
    # Build file paths
    input_file = Path(paths.input_path) / f"{date_str}_cps.parquet"
    output_diag_prefix = Path(paths.diagnostic_plots_path) / date_str
    output_stats_prefix = Path(paths.cal_stats_path) / date_str
    output_cal_file = Path(paths.output_path) / f"{date_str}_calibrated.csv"
    
    # Ensure output directories exist
    Path(paths.diagnostic_plots_path).mkdir(parents=True, exist_ok=True)
    Path(paths.cal_stats_path).mkdir(parents=True, exist_ok=True)
    Path(paths.output_path).mkdir(parents=True, exist_ok=True)
    
    # Load CPS data
    df = load_cps_data(str(input_file))
    outputs['cps_data'] = df
    
    # Check if calibration was performed (cal_flow has variation)
    if df['cal_flow'].std() <= 0.1:
        print("  No calibration detected (cal_flow has no variation)")
        
        # Still process zeros
        species_cols = list(df.columns[config.species_col_range[0]:config.species_col_range[1]])
        zero_df = process_in_drive_zeros(df, config, species_cols)
        
        if len(zero_df) > 0:
            zero_path = f"{output_stats_prefix}_zeros.csv"
            zero_df.to_csv(zero_path, index=False)
            print(f"  ✓ Wrote zeros: {zero_path}")
            outputs['zeros'] = zero_df
        
        return outputs
    
    # -------------------------------------------------------------------------
    # Load calibration standards
    # -------------------------------------------------------------------------
    print("\n→ Loading calibration standards...")
    cs_long = load_calibration_standards(paths.cal_standard_path)
    
    # Filter to current cylinder and positive concentrations
    cal_cyl_df = cs_long[
        (cs_long['cylinder_number'] == config.cal_cylinder_no) &
        (cs_long['cal_concs'] > 0)
    ][['species', 'cal_concs']].copy()
    
    if len(cal_cyl_df) == 0:
        raise ValueError(f"No calibration data found for cylinder {config.cal_cylinder_no}")
    
    # -------------------------------------------------------------------------
    # Identify species columns
    # -------------------------------------------------------------------------
    species_cols = list(df.columns[config.species_col_range[0]:config.species_col_range[1]])
    # Filter to columns that actually exist
    species_cols = [c for c in species_cols if c in df.columns]
    
    print(f"  Using {len(species_cols)} species columns")
    
    # -------------------------------------------------------------------------
    # Find calibration windows
    # -------------------------------------------------------------------------
    print("\n→ Finding calibration windows...")
    
    cal_windows = []
    
    # Zero step (cal_valve=0, zero_valve=1)
    zero_window = find_cal_window(df, 0, 0, 1, config, cal_cyl_df, species_cols)
    if zero_window is not None:
        print(f"    Found zero window: {len(zero_window)} species")
        cal_windows.append(zero_window)
    
    # Calibration steps (cal_valve=1, zero_valve=1)
    for flow in [f for f in config.cal_step_flows if f > 0]:
        window = find_cal_window(df, flow, 1, 1, config, cal_cyl_df, species_cols)
        if window is not None:
            print(f"    Found {flow} sccm window: {len(window)} species")
            cal_windows.append(window)
    
    if not cal_windows:
        raise ValueError("No calibration windows found")
    
    # Combine all windows
    stats_df = pd.concat(cal_windows, ignore_index=True)
    outputs['cal_windows'] = stats_df
    
    # -------------------------------------------------------------------------
    # Compute calibration statistics
    # -------------------------------------------------------------------------
    print("\n→ Computing calibration statistics...")
    results_df = compute_cal_stats(stats_df)
    
    # -------------------------------------------------------------------------
    # k_PTR sensitivity analysis
    # -------------------------------------------------------------------------
    print("\n→ Computing k_PTR sensitivity...")
    ksens_df = get_ksens_data(config.cal_cylinder_no, paths.ksens_standards_path)
    
    merged_df, ksens_value, plot_df = compute_ksens_relationship(
        results_df, ksens_df, config.default_k
    )
    
    # Add ksens to results
    results_df['ksens'] = ksens_value
    outputs['cal_stats'] = results_df
    
    # -------------------------------------------------------------------------
    # Generate diagnostic plots
    # -------------------------------------------------------------------------
    print("\n→ Generating diagnostic plots...")
    
    curves_path = f"{output_diag_prefix}_curves.pdf"
    plot_calibration_curves(stats_df, curves_path)
    
    if len(plot_df) > 0:
        exclude_species = ["CH5S", "CH5O", "C2H7O", "C10H23"]
        exclude_df = merged_df[merged_df['species'].isin(exclude_species)]
        ksens_path = f"{output_diag_prefix}_ksens.pdf"
        plot_ksens_relationship(plot_df, exclude_df, ksens_path)
    
    # -------------------------------------------------------------------------
    # Save calibration statistics
    # -------------------------------------------------------------------------
    print("\n→ Saving calibration statistics...")
    
    stats_path = f"{output_stats_prefix}_calstats.csv"
    results_df.to_csv(stats_path, index=False)
    print(f"  ✓ Wrote: {stats_path}")
    
    # -------------------------------------------------------------------------
    # Process in-drive zeros
    # -------------------------------------------------------------------------
    print("\n→ Processing in-drive zeros...")
    zero_df = process_in_drive_zeros(df, config, species_cols)
    
    zero_path = f"{output_stats_prefix}_zeros.csv"
    if len(zero_df) > 0:
        zero_df.to_csv(zero_path, index=False)
        outputs['zeros'] = zero_df
    else:
        pd.DataFrame().to_csv(zero_path, index=False)
    print(f"  ✓ Wrote: {zero_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Completed VOCUS calibrations for {date_str}")
    print(f"{'='*60}\n")
    
    return outputs


def process_vocus_calibrations_batch(
    date_list: List[str],
    config: VOCUSCalConfig,
    paths: VOCUSCalPathConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Process VOCUS calibrations for multiple dates.
    
    Parameters
    ----------
    date_list : list
        List of dates in YYYYMMDD format
    config : VOCUSCalConfig
        Calibration configuration
    paths : VOCUSCalPathConfig
        Input/output paths
    
    Returns
    -------
    dict
        Dictionary mapping date -> outputs
    """
    all_outputs = {}
    
    for date_str in date_list:
        try:
            outputs = process_vocus_calibrations(date_str, config, paths)
            all_outputs[date_str] = outputs
        except Exception as e:
            print(f"ERROR processing {date_str}: {e}")
            all_outputs[date_str] = {'error': str(e)}
    
    return all_outputs


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process VOCUS PTR-ToF calibrations"
    )
    parser.add_argument(
        'dates',
        nargs='+',
        help='Dates to process (YYYYMMDD format)'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory with CPS files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for calibrated files'
    )
    parser.add_argument(
        '--diagnostics', '-d',
        required=True,
        help='Directory for diagnostic plots'
    )
    parser.add_argument(
        '--cal-stats',
        required=True,
        help='Directory for calibration statistics'
    )
    parser.add_argument(
        '--standards',
        required=True,
        help='Path to calibration standards CSV'
    )
    parser.add_argument(
        '--cylinder',
        required=True,
        help='Calibration cylinder number'
    )
    parser.add_argument(
        '--ksens-standards',
        required=False,
        default=None,
        help='Path to k_PTR sensitivity standards CSV'
    )
    parser.add_argument(
        '--default-k',
        type=float,
        default=2.5,
        help='Default k_PTR value (default: 2.5)'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = VOCUSCalConfig(
        default_k=args.default_k,
        cal_cylinder_no=args.cylinder
    )
    
    paths = VOCUSCalPathConfig(
        input_path=args.input,
        output_path=args.output,
        diagnostic_plots_path=args.diagnostics,
        cal_stats_path=args.cal_stats,
        cal_standard_path=args.standards,
        ksens_standards_path=args.ksens_standards
    )
    
    # Run pipeline
    results = process_vocus_calibrations_batch(args.dates, config, paths)
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    for date_str, outputs in results.items():
        if 'error' in outputs:
            print(f"  {date_str}: FAILED - {outputs['error']}")
        elif 'cal_stats' in outputs:
            n_species = len(outputs['cal_stats'])
            print(f"  {date_str}: {n_species} species calibrated")
        else:
            print(f"  {date_str}: No calibration (zeros only)")