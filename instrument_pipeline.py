"""
Instrument Processing Pipeline for CalMAPLab data streams
================================================================

Complete pipeline for processing instrument data from CalMAPLab mobile platform:
1. Load raw drive data from VanDAQ and VOCUS
2. Apply instrument-specific lag corrections
3. Flag data based on QC thresholds
4. Join with processed GPS data
5. Apply calibrations (slope/intercept interpolation)
6. Generate Aclima-format output files (L1/L2) for SMMI project
7. Process target calibrations

Designed to integrate with the GPS processing pipeline (gps_pipeline.py).

Author: S. J. Cliff
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import re
from numba import njit

# Optional imports
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    warnings.warn("geopandas not available - spatial filtering disabled")

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    warnings.warn("h3 not available - H3 indexing disabled")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration parameters for the instrument processing pipeline."""
    
    # Organization and revision info
    org: str = "UCB"
    revision: str = "r1"
    output_types: List[str] = field(default_factory=lambda: ["L1", "L2a"])
    
    # Target calibration settings
    target_cylinders: List[str] = field(default_factory=list)
    target_instruments: List[str] = field(default_factory=list)
    vocus_target_flow: float = 1.0
    target_length: int = 60  # seconds
    
    # PTR-MS parameters (ions to keep)
    ptr_prefixes: str = "HCOKSNV"
    
    # Rounding precision for output
    value_precision: int = 5
    coord_precision: int = 5
    
    # Maximum interpolation gap for calibrations (seconds)
    max_cal_interp_gap_s: float = 86400 * 7  # 7 days
    
    def __post_init__(self):
        """Generate PTR regex pattern from prefixes."""
        self.ptr_regex = f"^[{self.ptr_prefixes}]"


@dataclass 
class PathConfig:
    """File paths for input/output data."""
    
    # Input paths
    drive_in: str = ""
    vocus_in: str = ""
    gps_in: str = ""
    
    # Calibration files
    NO_NO2_NOx_O3_cal_files: str = ""
    CO_N2O_CH4_C2H6_CO2_cal_files: str = ""
    vocus_cal_stats: str = ""
    
    # Reference files
    poly: str = ""  # GeoJSON for spatial filtering
    cylinders: str = ""  # Calibration cylinder concentrations
    flags_file: str = ""  # QC flag thresholds
    lag_times: str = ""  # Instrument lag times
    aclima_field: str = ""  # Field name mappings
    
    # Output paths
    aclima_out: str = ""
    drive_out: str = ""
    target_output: str = ""


# =============================================================================
# Part 1: Data Loading
# =============================================================================

def read_drive_data(filepath: str) -> pd.DataFrame:
    """
    Read raw drive data from VanDAQ long-format CSV.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file (measurements_van1_YYYY-MM-DD_*.csv)
    
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        [sample_time, instrument, parameter, value, string]
    """
    print(f"  Loading drive data: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Parse timestamp
    df['sample_time'] = pd.to_datetime(df['sample_time'], utc=True)
    
    # Keep only columns we need
    keep_cols = ['sample_time', 'instrument', 'parameter', 'value', 'string']
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # Ensure string column exists
    if 'string' not in df.columns:
        df['string'] = None
    
    print(f"    Loaded {len(df):,} rows, {df['instrument'].nunique()} instruments")
    
    return df


def read_vocus_data(filepath: str, timezone: str = "America/Los_Angeles") -> pd.DataFrame:
    """
    Read VOCUS PTR-ToF cps data and convert to long format.
    
    Parameters
    ----------
    filepath : str
        Path to VOCUS cps CSV file (YYYYMMDD_cps.csv)
    timezone : str
        Local timezone for timestamp conversion
    
    Returns
    -------
    pd.DataFrame
        Long-format dataframe matching VanDAQ structure
    """
    print(f"  Loading VOCUS data: {filepath}")
    
    df = pd.read_parquet(filepath)
    
    # Parse timestamp - VOCUS timestamps are in local time
    # Handle mixed formats (with/without microseconds)
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    
    # Convert to UTC (force local -> convert to UTC)
    df['sample_time'] = (
        df['date']
        .dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
        .dt.tz_convert('UTC')
        .dt.floor('s')
    )
    
    # Melt to long format
    id_vars = ['sample_time']
    value_vars = [c for c in df.columns if c not in ['date', 'sample_time']]
    
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='parameter',
        value_name='value'
    )
    
    df_long['instrument'] = 'Vocus-PTR-ToF'
    df_long['string'] = None
    
    # Reorder columns to match VanDAQ
    df_long = df_long[['sample_time', 'instrument', 'parameter', 'value', 'string']]
    
    print(f"    Loaded {len(df_long):,} rows, {df_long['parameter'].nunique()} parameters")
    
    return df_long


def load_drive_and_vocus(
    date_str: str,
    paths: PathConfig,
    date_format_change: str = "2025-06-29"
) -> pd.DataFrame:
    """
    Load and combine VanDAQ drive data with VOCUS data for a given date.
    
    Parameters
    ----------
    date_str : str
        Date string in YYYY-MM-DD format
    paths : PathConfig
        Configuration with input paths
    date_format_change : str
        Date after which filename format changed
    
    Returns
    -------
    pd.DataFrame
        Combined long-format dataframe
    """
    from pathlib import Path
    
    # Determine filename format based on date
    date_dt = pd.to_datetime(date_str)
    cutoff = pd.to_datetime(date_format_change)
    
    if date_dt > cutoff:
        drive_file = f"measurements_van1_{date_str}_drive_range_long.csv"
    else:
        drive_file = f"measurements_van1_{date_str}_no-geolocations_long.csv"
    
    drive_path = Path(paths.drive_in) / drive_file
    
    if not drive_path.exists():
        raise FileNotFoundError(f"Drive data not found: {drive_path}")
    
    drive_df = read_drive_data(str(drive_path))
    
    # Try to load VOCUS data
    vocus_file = f"{date_str.replace('-', '')}_cps.parquet"
    vocus_path = Path(paths.vocus_in) / vocus_file
    
    if vocus_path.exists():
        vocus_df = read_vocus_data(str(vocus_path))
        combined = pd.concat([drive_df, vocus_df], ignore_index=True)
        print(f"  Combined: {len(combined):,} total rows")
    else:
        print(f"  VOCUS file not found: {vocus_path}")
        combined = drive_df
    
    return combined


# =============================================================================
# Part 2: GPS Loading and Window
# =============================================================================

def read_processed_gps(filepath: str, timezone: str = "America/Los_Angeles") -> pd.DataFrame:
    """
    Read processed GPS data from RDS, Parquet, or CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to processed GPS file
    timezone : str
        Local timezone for timestamp conversion
    
    Returns
    -------
    pd.DataFrame
        GPS dataframe with sample_time in UTC
    """
    from pathlib import Path
    
    path = Path(filepath)
    print(f"  Loading GPS data: {filepath}")
    
    if path.suffix == '.parquet':
        gps = pd.read_parquet(filepath)
        # Rename datetime to sample_time if needed
        if 'datetime' in gps.columns and 'sample_time' not in gps.columns:
            gps = gps.rename(columns={'datetime': 'sample_time'})
    elif path.suffix == '.rds':
        # Try pyreadr first, fall back to rpy2 if available
        gps = None
        
        # Try pyreadr
        try:
            import pyreadr
            result = pyreadr.read_r(filepath)
            gps = list(result.values())[0]
        except Exception as e:
            print(f"    pyreadr failed: {e}")
        
        # Try rpy2 as fallback
        if gps is None:
            try:
                import rpy2.robjects as robjects
                from rpy2.robjects import pandas2ri
                pandas2ri.activate()
                
                readRDS = robjects.r['readRDS']
                r_df = readRDS(filepath)
                gps = pandas2ri.rpy2py(r_df)
            except Exception as e:
                print(f"    rpy2 failed: {e}")
        
        # If both fail, look for a CSV version
        if gps is None:
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                print(f"    Falling back to CSV: {csv_path}")
                gps = pd.read_csv(csv_path)
            else:
                raise ValueError(
                    f"Cannot read RDS file: {filepath}\n"
                    "Options:\n"
                    "  1. Export from R as CSV: write.csv(readRDS('file.rds'), 'file.csv')\n"
                    "  2. Export from R as Parquet: arrow::write_parquet(readRDS('file.rds'), 'file.parquet')\n"
                    "  3. Install rpy2 with R available: pip install rpy2"
                )
        
        if 'datetime' in gps.columns and 'sample_time' not in gps.columns:
            gps = gps.rename(columns={'datetime': 'sample_time'})
    else:
        # Assume CSV
        gps = pd.read_csv(filepath)
        if 'datetime' in gps.columns and 'sample_time' not in gps.columns:
            gps = gps.rename(columns={'datetime': 'sample_time'})
    
    # Parse and convert timestamp to UTC
    if not pd.api.types.is_datetime64_any_dtype(gps['sample_time']):
        gps['sample_time'] = pd.to_datetime(gps['sample_time'])
    
    # Check if timestamps are already in UTC
    # Parquet files from the GPS pipeline store UTC timestamps
    ts_tz = gps['sample_time'].dt.tz
    
    if ts_tz is None:
        # Timezone-naive - check if this looks like UTC already
        # (parquet files from gps_pipeline.py are UTC but stored naive)
        # Just localize to UTC directly without conversion
        print(f"    Timestamps are timezone-naive, assuming UTC")
        gps['sample_time'] = gps['sample_time'].dt.tz_localize('UTC')
    elif str(ts_tz) != 'UTC':
        # Has timezone but not UTC - convert
        print(f"    Converting from {ts_tz} to UTC")
        gps['sample_time'] = gps['sample_time'].dt.tz_convert('UTC')
    else:
        print(f"    Timestamps already in UTC")
    
    # Keep relevant columns
    keep_cols = [
        'sample_time', 'lat', 'lon', 'lon_smooth', 'lat_smooth',
        'lat_snap', 'lon_snap', 'lat_30', 'lon_30', 'drive_pass', 'rd_type'
    ]
    gps = gps[[c for c in keep_cols if c in gps.columns]]
    
    print(f"    Loaded {len(gps):,} GPS points")
    
    return gps


def gps_window(gps: pd.DataFrame) -> pd.DataFrame:
    """
    Get the first and last GPS timestamps (drive window).
    
    Parameters
    ----------
    gps : pd.DataFrame
        GPS dataframe with sample_time column
    
    Returns
    -------
    pd.DataFrame
        Two-row dataframe with first and last timestamps
    """
    gps_sorted = gps.sort_values('sample_time')
    return pd.concat([gps_sorted.iloc[[0]], gps_sorted.iloc[[-1]]])


# =============================================================================
# Part 3: Lag Correction
# =============================================================================

def load_lag_times(filepath: str) -> pd.DataFrame:
    """
    Load and reshape lag times from CSV.
    
    Expected format: columns for each instrument, rows by month.
    
    Parameters
    ----------
    filepath : str
        Path to lag times CSV
    
    Returns
    -------
    pd.DataFrame
        Long-format: [month, instrument, lag]
    """
    lag = pd.read_csv(filepath)
    
    # Check if already in long format
    if 'instrument' in lag.columns and 'lag' in lag.columns:
        lag['month'] = pd.to_datetime(lag['month'])
        return lag
    
    # Reshape from wide to long
    id_vars = ['month'] if 'month' in lag.columns else []
    value_vars = [c for c in lag.columns if c != 'month']
    
    lag_long = lag.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='instrument',
        value_name='lag'
    )
    
    # Parse month
    lag_long['month'] = pd.to_datetime(lag_long['month'], format='%m/%d/%Y')
    
    return lag_long


def apply_lag_correction(data: pd.DataFrame, lag_times: pd.DataFrame) -> pd.DataFrame:
    """
    Apply instrument-specific lag corrections to timestamps.
    
    Parameters
    ----------
    data : pd.DataFrame
        Drive data with sample_time and instrument columns
    lag_times : pd.DataFrame
        Lag times with month, instrument, lag columns
    
    Returns
    -------
    pd.DataFrame
        Data with lag-corrected sample_time
    """
    print("  Applying lag corrections...")
    
    data = data.copy()
    
    # Add month column for joining
    data['month'] = data['sample_time'].dt.to_period('M').dt.to_timestamp()
    lag_times = lag_times.copy()
    lag_times['month'] = lag_times['month'].dt.to_period('M').dt.to_timestamp()
    
    # Merge lag times
    data = data.merge(
        lag_times[['month', 'instrument', 'lag']],
        on=['month', 'instrument'],
        how='left'
    )
    
    # Apply lag correction
    data['lag'] = data['lag'].fillna(0)
    data['sample_time'] = data['sample_time'] - pd.to_timedelta(data['lag'], unit='s')
    
    # Clean up
    data = data.drop(columns=['lag', 'month'])
    
    n_corrected = (data['lag'] != 0).sum() if 'lag' in data.columns else 0
    print(f"    Applied lag to {n_corrected:,} rows")
    
    return data


# =============================================================================
# Part 4: Data Flagging
# =============================================================================

def load_flags(filepath: str) -> pd.DataFrame:
    """
    Load QC flag thresholds from CSV.
    
    Expected columns: instrument, parameter, range_start, range_end, flag_value
    
    Parameters
    ----------
    filepath : str
        Path to flags CSV
    
    Returns
    -------
    pd.DataFrame
        Flag thresholds
    """
    return pd.read_csv(filepath)


def flag_data(
    data: pd.DataFrame,
    gps_window: pd.DataFrame,
    flags: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply QC flags based on threshold criteria.
    
    Parameters
    ----------
    data : pd.DataFrame
        Lag-corrected drive data
    gps_window : pd.DataFrame
        First/last GPS timestamps (drive window)
    flags : pd.DataFrame
        Flag thresholds
    
    Returns
    -------
    pd.DataFrame
        Data with summary_flag column
    """
    print("  Applying QC flags...")
    
    data = data.copy()
    
    # Get GPS time window
    t_min = gps_window['sample_time'].min()
    t_max = gps_window['sample_time'].max()
    
    # Debug: print time ranges
    print(f"    GPS window: {t_min} to {t_max}")
    print(f"    Drive data range: {data['sample_time'].min()} to {data['sample_time'].max()}")
    
    # Ensure timezone compatibility - make both tz-naive or both tz-aware
    data_tz = data['sample_time'].dt.tz
    gps_tz = t_min.tzinfo if hasattr(t_min, 'tzinfo') else None
    
    if data_tz is not None and gps_tz is None:
        # Data is tz-aware, GPS is naive - make GPS tz-aware
        t_min = pd.Timestamp(t_min).tz_localize(data_tz)
        t_max = pd.Timestamp(t_max).tz_localize(data_tz)
    elif data_tz is None and gps_tz is not None:
        # Data is naive, GPS is tz-aware - make data times comparable
        t_min = t_min.tz_localize(None) if hasattr(t_min, 'tz_localize') else pd.Timestamp(t_min).tz_localize(None)
        t_max = t_max.tz_localize(None) if hasattr(t_max, 'tz_localize') else pd.Timestamp(t_max).tz_localize(None)
    elif data_tz is not None and gps_tz is not None and str(data_tz) != str(gps_tz):
        # Both tz-aware but different - convert GPS to data's timezone
        t_min = pd.Timestamp(t_min).tz_convert(data_tz)
        t_max = pd.Timestamp(t_max).tz_convert(data_tz)
    
    # Filter to GPS time window
    before_filter = len(data)
    data = data[(data['sample_time'] >= t_min) & (data['sample_time'] <= t_max)]
    print(f"    Rows after time filter: {len(data)} (from {before_filter})")
    
    if len(data) == 0:
        print("    WARNING: No data in GPS time window!")
        print(f"    Data timezone: {data_tz}, GPS timezone: {gps_tz}")
        # Return empty dataframe with expected columns
        data['summary_flag'] = '0'
        return data
    
    # Handle Status parameter - convert string to numeric
    if 'string' in data.columns:
        mask = (data['parameter'] == 'Status') & data['string'].notna()
        data.loc[mask, 'value'] = pd.to_numeric(data.loc[mask, 'string'], errors='coerce')
    
    # Merge flag thresholds
    data = data.merge(
        flags[['instrument', 'parameter', 'range_start', 'range_end', 'flag_value']],
        on=['instrument', 'parameter'],
        how='left'
    )
    
    # Calculate individual flags
    out_of_range = (
        data['range_start'].notna() & 
        data['range_end'].notna() &
        ((data['value'] < data['range_start']) | (data['value'] > data['range_end']))
    )
    data['flag'] = np.where(out_of_range, data['flag_value'].fillna(0).astype(int), 0)
    
    # Aggregate flags by sample_time and instrument
    summary_flags = (
        data.groupby(['sample_time', 'instrument'])['flag']
        .apply(lambda x: ','.join(map(str, sorted(set(x[x != 0])))) if (x != 0).any() else '0')
        .reset_index()
        .rename(columns={'flag': 'summary_flag'})
    )
    
    # Merge summary flags back
    data = data.drop(columns=['range_start', 'range_end', 'flag_value', 'flag'])
    data = data.merge(summary_flags, on=['sample_time', 'instrument'], how='left')
    
    # Drop string column and filter out non-CAPS NO2
    data = data.drop(columns=['string'], errors='ignore')
    data = data[
        data['parameter'].notna() &
        ~((data['parameter'] == 'NO2') & 
          (data['instrument'].isin(['EcoPhysics_NOX', 'Vocus-PTR-ToF'])))
    ]
    
    n_flagged = (data['summary_flag'] != '0').sum()
    print(f"    Flagged {n_flagged:,} rows")
    
    return data


# =============================================================================
# Part 5: GPS Join (Geolocation)
# =============================================================================

def join_gps(data: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """
    Join GPS data to instrument data by timestamp.
    
    Parameters
    ----------
    data : pd.DataFrame
        Flagged instrument data
    gps : pd.DataFrame
        Processed GPS data
    
    Returns
    -------
    pd.DataFrame
        Geolocated data
    """
    print("  Joining GPS to instrument data...")
    
    # Merge on sample_time (left join - keep all instrument data)
    merged = data.merge(gps, on='sample_time', how='left')
    
    # Filter to valid coordinates
    if 'lat_smooth' in merged.columns and 'lon_smooth' in merged.columns:
        merged = merged[merged['lat_smooth'].notna() & merged['lon_smooth'].notna()]
    elif 'lat' in merged.columns and 'lon' in merged.columns:
        merged = merged[merged['lat'].notna() & merged['lon'].notna()]
    
    # Sort
    merged = merged.sort_values(['sample_time', 'instrument', 'parameter'])
    
    print(f"    Joined: {len(merged):,} geolocated rows")
    
    return merged


# =============================================================================
# Part 6: Calibration
# =============================================================================

def interp_lin(
    t: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> np.ndarray:
    """
    Vectorized two-point linear interpolation.
    
    Parameters
    ----------
    t : array
        Target timestamps (numeric)
    t1 : array
        Start timestamps
    t2 : array
        End timestamps
    v1 : array
        Start values
    v2 : array
        End values
    
    Returns
    -------
    array
        Interpolated values
    """
    den = t2 - t1
    ok = np.isfinite(t1) & np.isfinite(t2) & np.isfinite(v1) & np.isfinite(v2)
    same = ok & (den == 0)
    
    result = np.full_like(t, np.nan, dtype=float)
    
    # Different endpoints - interpolate
    interp_mask = ok & ~same
    result[interp_mask] = (
        v1[interp_mask] + 
        (v2[interp_mask] - v1[interp_mask]) * 
        (t[interp_mask] - t1[interp_mask]) / den[interp_mask]
    )
    
    # Same endpoints - use v1
    result[same] = v1[same]
    
    return result


def load_calibration_stats(
    NO_NO2_NOx_O3_files: List[str],
    CO_N2O_CH4_C2H6_CO2_files: List[str],
    vocus_cal_files: List[str],
    vocus_zero_files: List[str],
    ptr_regex: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and harmonize all calibration statistics.
    
    Parameters
    ----------
    NO_NO2_NOx_O3_files : list
        Paths to NOx/O3 calibration stats
    CO_N2O_CH4_C2H6_CO2_files : list
        Paths to Aeris/LICOR calibration stats
    vocus_cal_files : list
        Paths to VOCUS slope stats
    vocus_zero_files : list
        Paths to VOCUS zero stats
    ptr_regex : str
        Regex pattern for PTR ion columns
    
    Returns
    -------
    tuple
        (slope_intercept_stats, vocus_zero_stats)
    """
    import re
    
    print("  Loading calibration statistics...")
    
    dfs = []
    
    # Load Aeris/LICOR stats
    for f in CO_N2O_CH4_C2H6_CO2_files:
        try:
            df = pd.read_csv(f)
            df['sample_time'] = pd.to_datetime(df['sample_time'], utc=True)
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not load {f}: {e}")
    
    # Load NOx/O3 stats
    for f in NO_NO2_NOx_O3_files:
        try:
            df = pd.read_csv(f)
            # Rename columns to match
            rename_map = {'compound': 'parameter', 'date': 'sample_time', 'instrument_d': 'instrument'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df['sample_time'] = pd.to_datetime(df['sample_time'], utc=True)
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not load {f}: {e}")
    
    # Load VOCUS cal stats
    v_cal_dfs = []
    for f in vocus_cal_files:
        try:
            df = pd.read_csv(f)
            
            # Parse date - handle mixed formats (some with Z suffix, some without)
            # UTC format: 2026-01-09T08:31:14.499858Z
            # Local format: 2025-04-12 07:27:51.499991
            df['sample_time'] = pd.to_datetime(df['date'], utc=True, format='mixed').dt.floor('s')
            
            df = df.rename(columns={'species': 'parameter'})
            df['instrument'] = 'Vocus-PTR-ToF'
            v_cal_dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not load {f}: {e}")
    
    # Load VOCUS zero stats
    v_zero_dfs = []
    for f in vocus_zero_files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
            # Remove unwanted columns
            df = df.drop(columns=['V1', 'flag'], errors='ignore')
            v_zero_dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not load {f}: {e}")
    
    # Combine slope/intercept stats
    if v_cal_dfs:
        v_cal = pd.concat(v_cal_dfs, ignore_index=True)
        
        # Get PTR columns from zero files for filling missing parameters
        if v_zero_dfs:
            v_zero_combined = pd.concat(v_zero_dfs, ignore_index=True)
            ptr_cols = [c for c in v_zero_combined.columns if re.match(ptr_regex, c)]
            
            # Fill missing params with ksens slope
            existing_params = v_cal['parameter'].unique()
            missing_params = [p for p in ptr_cols if p not in existing_params]
            
            if missing_params and 'ksens' in v_cal.columns:
                add_rows = []
                for _, row in v_cal[['sample_time', 'ksens']].drop_duplicates().iterrows():
                    for param in missing_params:
                        add_rows.append({
                            'parameter': param,
                            'sample_time': row['sample_time'],
                            'slope': row['ksens'],
                            'instrument': 'Vocus-PTR-ToF'
                        })
                if add_rows:
                    v_cal = pd.concat([v_cal, pd.DataFrame(add_rows)], ignore_index=True)
        
        dfs.append(v_cal[['instrument', 'parameter', 'sample_time', 'slope']].copy())
    
    # Combine all calibration stats
    if dfs:
        gas_cal_stats = pd.concat(dfs, ignore_index=True)
        
        # Standardize columns
        for col in ['instrument', 'parameter', 'sample_time', 'slope', 'intercept']:
            if col not in gas_cal_stats.columns:
                gas_cal_stats[col] = np.nan
        
        gas_cal_stats['slope'] = pd.to_numeric(gas_cal_stats['slope'], errors='coerce')
        gas_cal_stats['intercept'] = pd.to_numeric(gas_cal_stats['intercept'], errors='coerce')
    else:
        gas_cal_stats = pd.DataFrame(columns=['instrument', 'parameter', 'sample_time', 'slope', 'intercept'])
    
    # Process VOCUS zeros (wide to long)
    if v_zero_dfs:
        v_zero = pd.concat(v_zero_dfs, ignore_index=True)
        
        ptr_cols = [c for c in v_zero.columns if re.match(ptr_regex, c)]
        
        v_zero_long = v_zero.melt(
            id_vars=['date'],
            value_vars=ptr_cols,
            var_name='parameter',
            value_name='zero'
        )
        v_zero_long['instrument'] = 'Vocus-PTR-ToF'
        
        # Parse date - handle mixed formats (some with Z suffix, some without)
        # UTC format: 2026-01-09T08:31:14.499858Z
        # Local format: 2025-04-12 07:27:51.499991
        date_col = pd.to_datetime(v_zero_long['date'], utc=True, format='mixed')
        
        # Floor to seconds and ensure UTC
        v_zero_long['sample_time'] = date_col.dt.floor('s')
        
        v_zero_long = v_zero_long.drop(columns=['date'])
    else:
        v_zero_long = pd.DataFrame(columns=['parameter', 'instrument', 'sample_time', 'zero'])
    
    print(f"    Loaded {len(gas_cal_stats):,} calibration records, {len(v_zero_long):,} zero records")
    
    return gas_cal_stats, v_zero_long

@njit
def interp_lin_numba(t, t1, t2, v1, v2):
    """Vectorized linear interpolation with numba."""
    n = len(t)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        if t1[i] == t2[i]:
            result[i] = v1[i]
        elif np.isnan(t1[i]) or np.isnan(t2[i]):
            result[i] = np.nan
        else:
            frac = (t[i] - t1[i]) / (t2[i] - t1[i])
            result[i] = v1[i] + frac * (v2[i] - v1[i])
    return result


def calibrate_fast(
    data: pd.DataFrame,
    gas_cal_stats: pd.DataFrame,
    vocus_zeros: pd.DataFrame,
    ptr_regex: str
) -> pd.DataFrame:
    """
    Optimized calibration using vectorized operations.
    """
    print("  Applying calibrations...")
    
    data = data.copy()
    
    # Convert times once, use int64 directly
    t = data['sample_time'].values.astype('datetime64[s]').astype('int64')
    t_min, t_max = t.min(), t.max()
    
    n_rows = len(data)
    
    # Pre-allocate calibration arrays
    slope_i = np.ones(n_rows, dtype=np.float64)
    intercept_i = np.zeros(n_rows, dtype=np.float64)
    zero_i = np.zeros(n_rows, dtype=np.float64)
    
    # Build lookup indices for (instrument, parameter) pairs
    data['_idx'] = np.arange(n_rows)
    inst_param_groups = data.groupby(['instrument', 'parameter'])['_idx'].apply(np.array).to_dict()
    
    # Process gas calibrations if available
    if len(gas_cal_stats) > 0:
        gas_cal_stats = gas_cal_stats.copy()
        gas_cal_stats['t_cal'] = gas_cal_stats['sample_time'].values.astype('datetime64[s]').astype('int64')
        if 'intercept' not in gas_cal_stats.columns:
            gas_cal_stats['intercept'] = 0.0
        gas_cal_stats['intercept'] = gas_cal_stats['intercept'].fillna(0.0)
        
        # Group calibrations once
        cal_groups = gas_cal_stats.groupby(['instrument', 'parameter'])
        
        for (inst, param), group in cal_groups:
            if (inst, param) not in inst_param_groups:
                continue
            
            indices = inst_param_groups[(inst, param)]
            t_data = t[indices]
            
            group = group.sort_values('t_cal')
            t_cal = group['t_cal'].values
            slopes = group['slope'].values
            intercepts = group['intercept'].values
            
            # Find bracketing calibrations
            before_mask = t_cal < t_min
            after_mask = t_cal > t_max
            
            if before_mask.any() and after_mask.any():
                before_idx = np.where(before_mask)[0][-1]
                after_idx = np.where(after_mask)[0][0]
                t1, t2 = t_cal[before_idx], t_cal[after_idx]
                s1, s2 = slopes[before_idx], slopes[after_idx]
                i1, i2 = intercepts[before_idx], intercepts[after_idx]
            elif before_mask.any():
                idx = np.where(before_mask)[0][-1]
                t1 = t2 = t_cal[idx]
                s1 = s2 = slopes[idx]
                i1 = i2 = intercepts[idx]
            elif after_mask.any():
                idx = np.where(after_mask)[0][0]
                t1 = t2 = t_cal[idx]
                s1 = s2 = slopes[idx]
                i1 = i2 = intercepts[idx]
            else:
                continue
            
            # Vectorized interpolation for this group
            t1_arr = np.full(len(indices), t1)
            t2_arr = np.full(len(indices), t2)
            
            slope_i[indices] = interp_lin_numba(
                t_data, t1_arr, t2_arr,
                np.full(len(indices), s1), np.full(len(indices), s2)
            )
            intercept_i[indices] = interp_lin_numba(
                t_data, t1_arr, t2_arr,
                np.full(len(indices), i1), np.full(len(indices), i2)
            )
    
    # Adjust intercepts for specific parameters (vectorized)
    param_values = data['parameter'].values
    for param in ['CH4', 'N2O', 'CO2MFd']:
        mask = param_values == param
        intercept_i[mask] /= 1000
    
    co_mask = param_values == 'CO'
    intercept_i[co_mask] = 0
    
    # Handle VOCUS zeros
    if len(vocus_zeros) > 0:
        vocus_zeros = vocus_zeros.copy()
        vocus_zeros['t_zero'] = vocus_zeros['sample_time'].values.astype('datetime64[s]').astype('int64')
        
        # Pre-compile regex
        ptr_pattern = re.compile(ptr_regex)
        
        # Get VOCUS parameters that match regex
        vocus_params = vocus_zeros['parameter'].unique()
        matching_params = [p for p in vocus_params if ptr_pattern.match(p)]
        
        # Pre-group zeros
        zero_groups = {
            p: vocus_zeros[vocus_zeros['parameter'] == p].sort_values('t_zero')
            for p in matching_params
        }
        
        # Process VOCUS data
        vocus_mask = data['instrument'].values == 'Vocus-PTR-ToF'
        
        for param, param_zeros in zero_groups.items():
            if len(param_zeros) < 2:
                continue
            
            key = ('Vocus-PTR-ToF', param)
            if key not in inst_param_groups:
                continue
            
            indices = inst_param_groups[key]
            t_data = t[indices]
            
            z_t = param_zeros['t_zero'].values
            z_v = param_zeros['zero'].values
            
            # Efficient bracketing with searchsorted
            idx_right = np.searchsorted(z_t, t_data)
            idx_right = np.clip(idx_right, 1, len(z_t) - 1)
            idx_left = idx_right - 1
            
            zero_vals = interp_lin_numba(
                t_data, z_t[idx_left], z_t[idx_right], z_v[idx_left], z_v[idx_right]
            )
            zero_i[indices] = np.where(np.isfinite(zero_vals), zero_vals, 0)
    
    # Apply calibration (vectorized)
    values = data['value'].values
    calibrated = (values - zero_i - intercept_i) / slope_i
    
    # Handle NO2 derivation
    eco_mask = (data['instrument'].values == 'EcoPhysics_NOX') & \
               np.isin(data['parameter'].values, ['NO', 'NOx'])
    
    no2_rows = None
    if eco_mask.any():
        eco_data = data.loc[eco_mask, ['sample_time', 'instrument', 'parameter']].copy()
        eco_data['calibrated_value'] = calibrated[eco_mask]
        
        eco_wide = eco_data.pivot_table(
            index=['sample_time', 'instrument'],
            columns='parameter',
            values='calibrated_value',
            aggfunc='mean'
        ).reset_index()
        
        if 'NO' in eco_wide.columns and 'NOx' in eco_wide.columns:
            eco_wide['NO2'] = eco_wide['NOx'] - eco_wide['NO']
            
            no2_rows = eco_wide[['sample_time', 'instrument']].copy()
            no2_rows['parameter'] = 'NO2'
            no2_rows['value'] = eco_wide['NO2'].values
            
            # Get coordinate columns
            coord_cols = ['lat', 'lon', 'lat_smooth', 'lon_smooth', 'lat_snap', 'lon_snap',
                         'lat_30', 'lon_30', 'drive_pass', 'summary_flag', 'rd_type']
            available_coords = [c for c in coord_cols if c in data.columns]
            
            if available_coords:
                coord_data = data.loc[eco_mask, ['sample_time'] + available_coords].drop_duplicates('sample_time')
                no2_rows = no2_rows.merge(coord_data, on='sample_time', how='left')
    
    # Update values in place
    data['value'] = calibrated
    
    # Drop temp column
    data = data.drop(columns=['_idx'])
    
    # Append NO2 if generated
    if no2_rows is not None:
        data = pd.concat([data, no2_rows], ignore_index=True)
    
    # Remove duplicates
    data = data.drop_duplicates(subset=['sample_time', 'instrument', 'parameter'])
    
    print(f"    Calibrated {len(data):,} rows")
    
    return data


# =============================================================================
# Part 7: Output Generation
# =============================================================================

def generate_aclima_l1(
    data: pd.DataFrame,
    aclima_fields: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate Aclima L1 format output (uncalibrated).
    
    Parameters
    ----------
    data : pd.DataFrame
        Geolocated instrument data (before calibration)
    aclima_fields : pd.DataFrame
        Field name mapping dictionary
    
    Returns
    -------
    pd.DataFrame
        Aclima L1 format
    """
    data = data.copy()
    
    # Add required columns
    data['status_indicator'] = (data['summary_flag'] != '0').astype(int)
    data['timestamp'] = data['sample_time']
    data['qualifier_code'] = data['summary_flag']
    data['latitude'] = data['lat']
    data['longitude'] = data['lon']
    data['parameter_ucb'] = data['parameter']
    
    # Merge with field mappings
    merged = data.merge(
        aclima_fields,
        left_on='parameter_ucb',
        right_on='parameter_ucb',
        how='inner'
    )
    
    # Select and rename columns
    output_cols = ['timestamp', 'parameter', 'method', 'duration',
                   'value', 'unit', 'latitude', 'longitude',
                   'status_indicator', 'qualifier_code']
    
    return merged[[c for c in output_cols if c in merged.columns]]


def generate_aclima_l2(
    data: pd.DataFrame,
    aclima_fields: pd.DataFrame,
    poly: Optional[Any] = None
) -> pd.DataFrame:
    """
    Generate Aclima L2 format output (calibrated).
    
    Parameters
    ----------
    data : pd.DataFrame
        Calibrated, geolocated instrument data
    aclima_fields : pd.DataFrame
        Field name mapping dictionary
    poly : GeoDataFrame, optional
        Polygons to filter out (exclusion zones)
    
    Returns
    -------
    pd.DataFrame
        Aclima L2 format
    """
    data = data.copy()
    
    # Add required columns
    data['status_indicator'] = (data['summary_flag'] != '0').astype(int)
    data['vehicle_id'] = 'UCB'
    
    # Unit conversions: ppb to ppm
    ppb_to_ppm = data['parameter'].isin(['CH4', 'N2O', 'CO'])
    h2o_licor = (data['parameter'] == 'H2O') & (data['instrument'] == 'Licor_7200_CO2')
    data.loc[ppb_to_ppm | h2o_licor, 'value'] *= 1000
    
    # ng to ug for BC
    data.loc[data['parameter'] == 'BC6', 'value'] /= 1000
    
    # Rename columns
    rename_map = {
        'sample_time': 'timestamp',
        'summary_flag': 'qualifier_codes',
        'lat_smooth': 'latitude',
        'lon_smooth': 'longitude',
        'parameter': 'parameter_ucb'
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
    
    # Merge with field mappings
    merged = data.merge(
        aclima_fields,
        on=['parameter_ucb', 'instrument'],
        how='inner'
    )
    
    # Add H3 index
    if HAS_H3 and 'latitude' in merged.columns and 'longitude' in merged.columns:
        valid = merged['latitude'].notna() & merged['longitude'].notna()
        merged.loc[valid, 'h3_index'] = merged.loc[valid].apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 15),
            axis=1
        )
    else:
        merged['h3_index'] = None
    
    # Spatial filtering (remove points inside exclusion polygons)
    if poly is not None and HAS_GEOPANDAS:
        merged_gdf = gpd.GeoDataFrame(
            merged,
            geometry=gpd.points_from_xy(merged['longitude'], merged['latitude']),
            crs='EPSG:4326'
        )
        
        # Find points inside polygons
        inside = merged_gdf.geometry.apply(lambda p: poly.contains(p).any())
        merged = merged[~inside]
    
    # Select output columns
    output_cols = ['timestamp', 'parameter', 'method', 'duration',
                   'value', 'unit', 'latitude', 'longitude', 'h3_index',
                   'status_indicator', 'qualifier_codes', 'vehicle_id']
    
    return merged[[c for c in output_cols if c in merged.columns]]


def round_output(data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Round numeric columns for file size reduction.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to round
    config : PipelineConfig
        Configuration with precision settings
    
    Returns
    -------
    pd.DataFrame
        Rounded data
    """
    data = data.copy()
    
    # Round value column
    if 'value' in data.columns:
        data['value'] = data['value'].round(config.value_precision)
    
    # Round coordinate columns
    coord_cols = ['lat', 'lon', 'lat_smooth', 'lon_smooth', 
                  'lat_snap', 'lon_snap', 'lat_30', 'lon_30',
                  'latitude', 'longitude']
    for col in coord_cols:
        if col in data.columns:
            data[col] = data[col].round(config.coord_precision)
    
    return data


# =============================================================================
# Part 8: Target Processing
# =============================================================================

def process_targets(
    data: pd.DataFrame,
    instruments: List[str],
    cylinder_concs: pd.DataFrame,
    target_cyls: List[str],
    vocus_target_flow: float = 1.0,
    target_length: int = 60
) -> pd.DataFrame:
    """
    Process in-drive target calibrations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Calibrated drive data
    instruments : list
        Instruments used for targets
    cylinder_concs : pd.DataFrame
        Cylinder concentration data
    target_cyls : list
        Cylinder IDs for targets
    vocus_target_flow : float
        Expected VOCUS cal flow rate
    target_length : int
        Seconds of target data to extract
    
    Returns
    -------
    pd.DataFrame
        Target calibration data
    """
    print("  Processing target calibrations...")
    
    # Build cylinder request mapping
    cyl_req = pd.DataFrame({
        'instrument': instruments,
        'cylinder_number': target_cyls
    })
    
    # Reshape cylinder concentrations to long format
    id_cols = list(cylinder_concs.columns[:2])  # First two columns are ID columns
    value_cols = list(cylinder_concs.columns[2:])
    
    cs_long = cylinder_concs.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='species_full',
        value_name='cal_conc'
    )
    
    # Extract species name (remove units suffix)
    cs_long['parameter'] = cs_long['species_full'].str[:-4]  # Remove last 4 chars (e.g., "_ppb")
    
    # Join with cylinder requirements
    cs_f = cs_long.merge(cyl_req, on='cylinder_number', how='inner')
    cs_f = cs_f[cs_f['cal_conc'] > 0]
    
    # Unit conversion for specific species
    cs_f.loc[cs_f['parameter'].isin(['CH4', 'CO', 'N2O']), 'cal_conc'] /= 1000
    
    # Filter data to target instruments
    data = data.copy()
    f = data[data['instrument'].isin(instruments)][['sample_time', 'instrument', 'parameter', 'value']].copy()
    
    # Gating parameters
    need_params = ['cal_valve', 'zero_valve', 'cal_flow', 'zero_flow', 'Inlet_Number']
    
    # Pivot gating parameters to wide format
    gate_data = f[f['parameter'].isin(need_params)].copy()
    
    if len(gate_data) == 0:
        print("    No gating parameters found")
        return pd.DataFrame(columns=['sample_time', 'instrument', 'parameter', 'val'])
    
    gate_wide = gate_data.pivot_table(
        index=['instrument', 'sample_time'],
        columns='parameter',
        values='value',
        aggfunc='last'
    ).reset_index()
    
    results = []
    
    # VOCUS targets
    if 'Vocus-PTR-ToF' in instruments:
        vocus_cols = ['cal_valve', 'zero_valve', 'cal_flow', 'zero_flow']
        if all(c in gate_wide.columns for c in vocus_cols):
            vocus_gate = gate_wide[
                (gate_wide['instrument'] == 'Vocus-PTR-ToF') &
                (gate_wide['cal_valve'] == 1) &
                (gate_wide['zero_valve'] == 1) &
                gate_wide['cal_flow'].notna() &
                gate_wide['zero_flow'].notna() &
                (gate_wide['cal_flow'] > vocus_target_flow - 0.5) &
                (gate_wide['cal_flow'] < vocus_target_flow + 0.5)
            ][['instrument', 'sample_time', 'cal_flow', 'zero_flow']]
            
            if len(vocus_gate) > 0:
                # Join gating rows to VOCUS data
                vocus_dt = f[f['instrument'] == 'Vocus-PTR-ToF'].merge(
                    vocus_gate, on=['instrument', 'sample_time']
                )
                
                # Identify periods
                vocus_dt = _make_periods(vocus_dt, ['instrument', 'parameter'])
                
                # Keep last target_length + 10 to -10 seconds of each period
                vocus_dt['t_end'] = vocus_dt.groupby(['instrument', 'parameter', 'period'])['sample_time'].transform('max')
                vocus_keep = vocus_dt[
                    (vocus_dt['sample_time'] >= vocus_dt['t_end'] - pd.Timedelta(seconds=target_length + 10)) &
                    (vocus_dt['sample_time'] <= vocus_dt['t_end'] - pd.Timedelta(seconds=10))
                ].copy()
                
                # Join cal concentrations
                vocus_keep = vocus_keep.merge(
                    cs_f[['instrument', 'parameter', 'cal_conc']],
                    on=['instrument', 'parameter'],
                    how='inner'
                )
                
                # Calculate dilution-corrected value
                vocus_keep['frac'] = np.where(
                    vocus_keep['cal_flow'] + vocus_keep['zero_flow'] > 0,
                    vocus_keep['cal_flow'] / (vocus_keep['cal_flow'] + vocus_keep['zero_flow']),
                    np.nan
                )
                vocus_keep['val'] = vocus_keep['value'] - (vocus_keep['cal_conc'] * vocus_keep['frac'])
                
                results.append(vocus_keep[['sample_time', 'instrument', 'parameter', 'val']])
    
    # Aeris targets
    aeris_insts = ['Aeris_N2O_CO', 'Aeris_CH4_C2H6']
    aeris_in_instruments = [i for i in aeris_insts if i in instruments]
    
    if aeris_in_instruments and 'Inlet_Number' in gate_wide.columns:
        aeris_inlet = gate_wide[
            (gate_wide['instrument'].isin(aeris_in_instruments)) &
            (gate_wide['Inlet_Number'] == 1)
        ][['instrument', 'sample_time']]
        
        if len(aeris_inlet) > 0:
            # Identify periods
            aeris_inlet = _make_periods(aeris_inlet, ['instrument'])
            
            # Get period end times
            aeris_periods = aeris_inlet.groupby(['instrument', 'period']).agg(
                t_end=('sample_time', 'max')
            ).reset_index()
            aeris_periods['t_win_start'] = aeris_periods['t_end'] - pd.Timedelta(seconds=70)
            aeris_periods['t_win_end'] = aeris_periods['t_end'] - pd.Timedelta(seconds=10)
            
            # Get all parameters within windows
            aeris_all = f[f['instrument'].isin(aeris_in_instruments)].copy()
            
            # Non-equi join (sample_time within window)
            aeris_keep = []
            for _, period_row in aeris_periods.iterrows():
                mask = (
                    (aeris_all['instrument'] == period_row['instrument']) &
                    (aeris_all['sample_time'] >= period_row['t_win_start']) &
                    (aeris_all['sample_time'] <= period_row['t_win_end'])
                )
                aeris_keep.append(aeris_all[mask])
            
            if aeris_keep:
                aeris_keep = pd.concat(aeris_keep, ignore_index=True)
                
                # Join cal concentrations
                aeris_keep = aeris_keep.merge(
                    cs_f[['instrument', 'parameter', 'cal_conc']],
                    on=['instrument', 'parameter'],
                    how='inner'
                )
                
                # Calculate residual
                aeris_keep['val'] = aeris_keep['value'] - aeris_keep['cal_conc']
                
                results.append(aeris_keep[['sample_time', 'instrument', 'parameter', 'val']])
    
    if results:
        output = pd.concat(results, ignore_index=True)
        output = output[output['val'].notna()]
        print(f"    Processed {len(output):,} target measurements")
        return output
    else:
        print("    No target data found")
        return pd.DataFrame(columns=['sample_time', 'instrument', 'parameter', 'val'])


def _make_periods(df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
    """
    Identify consecutive periods based on time gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with sample_time column
    by_cols : list
        Columns to group by
    
    Returns
    -------
    pd.DataFrame
        Data with 'period' column added
    """
    df = df.sort_values('sample_time').copy()
    
    # Calculate gaps
    df['gap'] = df.groupby(by_cols)['sample_time'].diff().dt.total_seconds()
    
    # New period when gap >= 3 seconds
    df['period'] = df.groupby(by_cols)['gap'].transform(
        lambda x: (x.isna() | (x >= 3)).cumsum()
    )
    
    df = df.drop(columns=['gap'])
    
    return df


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    date_str: str,
    config: PipelineConfig,
    paths: PathConfig
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete instrument processing pipeline for a single date.
    
    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    config : PipelineConfig
        Pipeline configuration
    paths : PathConfig
        Input/output paths
    
    Returns
    -------
    dict
        Dictionary of output dataframes
    """
    from pathlib import Path
    import glob
    
    print(f"\n{'='*60}")
    print(f"Processing {date_str}")
    print(f"{'='*60}")
    
    outputs = {}
    
    # -------------------------------------------------------------------------
    # Load static resources
    # -------------------------------------------------------------------------
    print("\n→ Loading static resources...")
    
    aclima_fields = pd.read_csv(paths.aclima_field)
    flags = load_flags(paths.flags_file)
    lag_times = load_lag_times(paths.lag_times)
    
    cylinder_concs = None
    if paths.cylinders and Path(paths.cylinders).exists():
        cylinder_concs = pd.read_csv(paths.cylinders)
    
    # Load calibration files
    NO_NO2_NOx_O3_files = glob.glob(str(Path(paths.NO_NO2_NOx_O3_cal_files) / "*"))
    CO_N2O_files = glob.glob(str(Path(paths.CO_N2O_CH4_C2H6_CO2_cal_files) / "*"))
    vocus_cal_files = glob.glob(str(Path(paths.vocus_cal_stats) / "*calstats*"))
    vocus_zero_files = glob.glob(str(Path(paths.vocus_cal_stats) / "*zero*"))
    
    gas_cal_stats, vocus_zeros = load_calibration_stats(
        NO_NO2_NOx_O3_files,
        CO_N2O_files,
        vocus_cal_files,
        vocus_zero_files,
        config.ptr_regex
    )
    
    # Load exclusion polygons
    poly = None
    if paths.poly and Path(paths.poly).exists() and HAS_GEOPANDAS:
        poly = gpd.read_file(paths.poly)
    
    # -------------------------------------------------------------------------
    # Load drive and VOCUS data
    # -------------------------------------------------------------------------
    print("\n→ Loading drive data...")
    drive_data = load_drive_and_vocus(date_str, paths)
    
    # -------------------------------------------------------------------------
    # Load GPS data
    # -------------------------------------------------------------------------
    print("\n→ Loading GPS data...")
    gps_pattern = f"processed_gps_{date_str}_complete*.rds"
    gps_files = glob.glob(str(Path(paths.gps_in) / gps_pattern))
    
    # Also try parquet
    if not gps_files:
        gps_pattern = f"processed_gps_{date_str}.parquet"
        gps_files = glob.glob(str(Path(paths.gps_in) / gps_pattern))
    
    if not gps_files:
        raise FileNotFoundError(f"No GPS file found matching {gps_pattern}")
    
    gps = read_processed_gps(gps_files[0])
    start_end_gps = gps_window(gps)
    
    # -------------------------------------------------------------------------
    # Apply lag correction
    # -------------------------------------------------------------------------
    print("\n→ Applying lag correction...")
    drive_lag = apply_lag_correction(drive_data, lag_times)
    
    # -------------------------------------------------------------------------
    # Apply flags
    # -------------------------------------------------------------------------
    print("\n→ Flagging data...")
    drive_flagged = flag_data(drive_lag, start_end_gps, flags)
    
    # -------------------------------------------------------------------------
    # Join GPS
    # -------------------------------------------------------------------------
    print("\n→ Joining GPS...")
    drive_geo = join_gps(drive_flagged, gps)
    
    # -------------------------------------------------------------------------
    # Calibrate
    # -------------------------------------------------------------------------
    print("\n→ Calibrating...")
    drive_cal = calibrate_fast(drive_geo, gas_cal_stats, vocus_zeros, config.ptr_regex)
    
    # Remove duplicates
    drive_cal = drive_cal.drop_duplicates(
        subset=['sample_time', 'instrument', 'parameter']
    )
    
    # Round values
    drive_cal = round_output(drive_cal, config)
    
    # Save for targets
    fortargets = drive_cal.copy()
    
    # -------------------------------------------------------------------------
    # Generate outputs
    # -------------------------------------------------------------------------
    print("\n→ Writing output files...")
    
    date_compact = date_str.replace('-', '')
    
    # UCB complete file
    out_name = f"{config.org}_complete_{date_compact}_L2a_{config.revision}.parquet"
    out_path = Path(paths.drive_out) / out_name
    drive_cal.to_parquet(out_path, index=False)
    print(f"  ✓ Wrote 1Hz drive file: {out_name}")
    outputs['ucb_complete'] = drive_cal
    
    # Aclima L2 file
    if 'L2a' in config.output_types:
        aclima_l2 = generate_aclima_l2(drive_cal, aclima_fields, poly)
        aclima_name = f"{config.org}_{date_compact}_L2a_{config.revision}.parquet"
        aclima_path = Path(paths.aclima_out) / aclima_name
        aclima_l2.to_parquet(aclima_path, index=False)
        print(f"  ✓ Wrote Aclima L2a file: {aclima_name}")
        outputs['aclima_l2'] = aclima_l2
    
    # -------------------------------------------------------------------------
    # Process targets
    # -------------------------------------------------------------------------
    if config.target_length > 0 and cylinder_concs is not None:
        print("\n→ Processing targets...")
        target_output = process_targets(
            fortargets,
            config.target_instruments,
            cylinder_concs,
            config.target_cylinders,
            config.vocus_target_flow,
            config.target_length
        )
        
        if len(target_output) > 0:
            target_name = f"{date_compact}_targets.parquet"
            target_path = Path(paths.target_output) / target_name
            target_output.to_parquet(target_path, index=False)
            print(f"  ✓ Wrote target file: {target_name}")
            outputs['targets'] = target_output
    
    print(f"\n{'='*60}")
    print(f"✓ Completed {date_str}")
    print(f"{'='*60}\n")
    
    return outputs


def run_pipeline_batch(
    date_list: List[str],
    config: PipelineConfig,
    paths: PathConfig
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the pipeline for multiple dates.
    
    Parameters
    ----------
    date_list : list
        List of dates in YYYY-MM-DD format
    config : PipelineConfig
        Pipeline configuration
    paths : PathConfig
        Input/output paths
    
    Returns
    -------
    dict
        Dictionary mapping date -> output dataframes
    """
    all_outputs = {}
    
    for date_str in date_list:
        try:
            outputs = run_pipeline(date_str, config, paths)
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
        description="Process CalMAPLab mobile instrument data"
    )
    parser.add_argument('dates', nargs='+', help='Dates to process (YYYY-MM-DD)')
    parser.add_argument('--config', help='Path to config YAML file')
    parser.add_argument('--drive-in', help='Path to raw drive data')
    parser.add_argument('--vocus-in', help='Path to VOCUS data')
    parser.add_argument('--gps-in', help='Path to processed GPS data')
    parser.add_argument('--output', help='Output directory')
    
    args = parser.parse_args()
    
    # Build config from args or defaults
    config = PipelineConfig()
    paths = PathConfig()
    
    if args.drive_in:
        paths.drive_in = args.drive_in
    if args.vocus_in:
        paths.vocus_in = args.vocus_in
    if args.gps_in:
        paths.gps_in = args.gps_in
    if args.output:
        paths.drive_out = args.output
        paths.aclima_out = args.output
        paths.target_output = args.output
    
    # Run pipeline
    run_pipeline_batch(args.dates, config, paths)