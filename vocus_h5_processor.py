"""
VOCUS H5 Processing Pipeline
============================

Processing code for converting VOCUS PTR-ToF HR peak fitted files .h5 files to a single parquet file of time and cps.

This module handles:
1. Loading raw VOCUS .h5 files (peak data and TPS data)
2. Converting counts to counts-per-second (cps)
3. Extracting timestamps and valve states
4. Generating processed parquet files for downstream analysis

Designed to integrate with the instrument processing pipeline (instrument_pipeline.py).

Author: S. J. Cliff
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import warnings
import glob
import re

# Optional imports
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not available - VOCUS H5 processing disabled")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VOCUSConfig:
    """Configuration parameters for VOCUS H5 processing."""
    
    # TOF extraction frequency (Hz)
    tof_extraction_freq: float = 24990.0


@dataclass
class VOCUSPathConfig:
    """File paths for VOCUS H5 processing."""
    
    # Input directory containing raw .h5 files
    input_base: str = ""
    
    # Output directory for processed CSV files
    output_base: str = ""
    
    # Path to peak columns CSV (ion species names)
    peak_columns_path: str = ""
    
    # Path to TPS columns CSV (valve states, flows, etc.)
    tps_columns_path: str = ""


# =============================================================================
# Column Configuration Loading
# =============================================================================

def load_peak_columns(filepath: str) -> List[str]:
    """
    Load peak column names from CSV file.
    
    Expected CSV columns: index, species, description
    
    Parameters
    ----------
    filepath : str
        Path to peak columns CSV
    
    Returns
    -------
    list
        List of species column names in order
    """
    print(f"  Loading peak columns: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Sort by index to ensure correct order
    df = df.sort_values('index')
    
    species_list = df['species'].tolist()
    
    print(f"    Loaded {len(species_list)} peak column names")
    
    return species_list


def load_tps_columns(filepath: str) -> Tuple[List[int], List[str]]:
    """
    Load TPS column indices and names from CSV file.
    
    Expected CSV columns: index, name, description
    
    Parameters
    ----------
    filepath : str
        Path to TPS columns CSV
    
    Returns
    -------
    tuple
        (list of indices, list of names)
    """
    print(f"  Loading TPS columns: {filepath}")
    
    df = pd.read_csv(filepath)
    
    indices = df['index'].tolist()
    names = df['name'].tolist()
    
    print(f"    Loaded {len(indices)} TPS columns")
    
    return indices, names


def get_column_config(paths: VOCUSPathConfig) -> Tuple[List[str], List[int], List[str]]:
    """
    Get column configuration from CSV files.
    
    Parameters
    ----------
    paths : VOCUSPathConfig
        Path configuration with peak_columns_path and tps_columns_path
    
    Returns
    -------
    tuple
        (peak_colnames, tps_indices, tps_colnames)
    
    Raises
    ------
    FileNotFoundError
        If required CSV files are not found
    """
    # Load peak columns
    if not paths.peak_columns_path:
        raise ValueError("peak_columns_path must be specified in VOCUSPathConfig")
    
    if not Path(paths.peak_columns_path).exists():
        raise FileNotFoundError(f"Peak columns file not found: {paths.peak_columns_path}")
    
    peak_colnames = load_peak_columns(paths.peak_columns_path)
    
    # Load TPS columns
    if not paths.tps_columns_path:
        raise ValueError("tps_columns_path must be specified in VOCUSPathConfig")
    
    if not Path(paths.tps_columns_path).exists():
        raise FileNotFoundError(f"TPS columns file not found: {paths.tps_columns_path}")
    
    tps_indices, tps_colnames = load_tps_columns(paths.tps_columns_path)
    
    return peak_colnames, tps_indices, tps_colnames


# =============================================================================
# Part 1: H5 File Loading
# =============================================================================

def _parse_h5_timestamp(filename: str) -> datetime:
    """
    Parse timestamp from VOCUS H5 filename.
    
    Expected formats:
    - YYYYMMDD_HHMMSS_p.h5
    - mobilelabYYYYMMDD_HHMMSS_p.h5
    - [prefix]YYYYMMDD_HHMMSS_p.h5
    
    Parameters
    ----------
    filename : str
        H5 filename (basename only)
    
    Returns
    -------
    datetime
        Parsed datetime object
    """
    # Extract datetime portion from filename
    basename = Path(filename).stem  # Remove .h5
    
    # Handle path separators
    if '\\' in basename:
        basename = basename.split('\\')[-1]
    if '/' in basename:
        basename = basename.split('/')[-1]
    
    # Remove _p suffix
    basename = basename.replace('_p', '')
    
    # Try to find YYYYMMDD_HHMMSS pattern anywhere in the string
    # This handles prefixes like 'mobilelab', 'van1', etc.
    pattern = r'(\d{8})_(\d{6})'
    match = re.search(pattern, basename)
    
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        datetime_str = f"{date_part}_{time_part}"
        return datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
    
    # Fallback: try parsing the whole string (for backwards compatibility)
    try:
        return datetime.strptime(basename, '%Y%m%d_%H%M%S')
    except ValueError:
        raise ValueError(
            f"Could not parse timestamp from filename: {filename}\n"
            f"Expected format: [prefix]YYYYMMDD_HHMMSS_p.h5"
        )


def load_peak_data(
    filepath: str,
    config: VOCUSConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load peak data (counts) from a single VOCUS H5 file.
    
    Parameters
    ----------
    filepath : str
        Path to the processed H5 file (*_p.h5)
    config : VOCUSConfig
        Configuration with TOF extraction frequency
    
    Returns
    -------
    tuple
        (timestamps, counts_per_second) as numpy arrays
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for VOCUS H5 processing")
    
    with h5py.File(filepath, 'r') as file:
        # Load peak data (counts per extraction)
        peakdata = file['/PeakData/PeakData']
        peakdata_arr = peakdata[:]
        counts = peakdata_arr.reshape(-1, peakdata_arr.shape[-1])
        
        # Convert to counts per second
        counts_cps = counts * config.tof_extraction_freq
        
        # Generate timestamps
        basetime = _parse_h5_timestamp(filepath)
        epoch = datetime(1970, 1, 1)
        basetime_unix = (basetime - epoch).total_seconds()
        
        # Load timing data
        tdata = file['/TimingData/BufTimes']
        times = tdata[:]
        secs = times.flatten()
        
        # Generate timestamp strings
        timestamps = np.array([
            datetime.utcfromtimestamp(basetime_unix + s).strftime('%Y-%m-%d %H:%M:%S.%f')
            for s in secs
        ])
        
        # Handle mismatched array lengths
        rows_counts = counts_cps.shape[0]
        rows_times = len(timestamps)
        
        if rows_times > rows_counts:
            # Pad counts with NaN
            missing = rows_times - rows_counts
            filler = np.full((missing, counts_cps.shape[1]), np.nan)
            counts_cps = np.vstack([counts_cps, filler])
        elif rows_counts > rows_times:
            # Truncate counts
            counts_cps = counts_cps[:rows_times]
        
        return timestamps, counts_cps


def load_tps_data(filepath: str, tps_indices: List[int]) -> np.ndarray:
    """
    Load TPS data (valve states, flows, etc.) from a VOCUS H5 file.
    
    Parameters
    ----------
    filepath : str
        Path to the raw H5 file (not *_p.h5)
    tps_indices : list
        List of column indices to extract
    
    Returns
    -------
    np.ndarray
        TPS data array with selected columns
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for VOCUS H5 processing")
    
    with h5py.File(filepath, 'r') as file:
        tps = file['/TPS2/TwData']
        tps_arr = tps[:]
        tps_data = tps_arr.reshape(-1, tps_arr.shape[-1])
        
        # Select only the columns we need
        tps_selected = tps_data[:, tps_indices]
        
        return tps_selected


# =============================================================================
# Part 2: Batch Processing
# =============================================================================

def process_day(
    date_str: str,
    paths: VOCUSPathConfig,
    config: Optional[VOCUSConfig] = None
) -> pd.DataFrame:
    """
    Process all VOCUS H5 files for a single day.
    
    Parameters
    ----------
    date_str : str
        Date string in YYYYMMDD format
    paths : VOCUSPathConfig
        Configuration with input/output paths
    config : VOCUSConfig, optional
        Processing configuration (uses defaults if not provided)
    
    Returns
    -------
    pd.DataFrame
        Combined and processed VOCUS data
    """
    if config is None:
        config = VOCUSConfig()
    
    if not HAS_H5PY:
        raise ImportError("h5py is required for VOCUS H5 processing")
    
    print(f"\n{'='*60}")
    print(f"Processing VOCUS data for {date_str}")
    print(f"{'='*60}")
    
    # Load column configuration
    print("\n→ Loading column configuration...")
    peak_colnames, tps_indices, tps_colnames = get_column_config(paths)
    
    # Set up paths
    input_dir = Path(paths.input_base) / date_str
    output_dir = Path(paths.output_base)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all processed H5 files
    peak_files = sorted(glob.glob(str(input_dir / "Processed" / "*_p.h5")))
    
    if not peak_files:
        print(f"  WARNING: No processed H5 files found in {input_dir / 'Processed'}")
        return pd.DataFrame()
    
    print(f"\n→ Found {len(peak_files)} peak data files")
    
    # Load and combine peak data
    print("  Loading peak data...")
    all_timestamps = []
    all_counts = []
    
    for filepath in peak_files:
        try:
            timestamps, counts = load_peak_data(filepath, config)
            all_timestamps.append(timestamps)
            all_counts.append(counts)
        except Exception as e:
            print(f"    WARNING: Could not load {Path(filepath).name}: {e}")
    
    if not all_counts:
        print("  ERROR: No peak data loaded")
        return pd.DataFrame()
    
    combined_timestamps = np.concatenate(all_timestamps)
    combined_counts = np.concatenate(all_counts)
    
    print(f"    Loaded {len(combined_timestamps):,} records from peak data")
    
    # Find and load TPS files
    print("\n→ Loading TPS data...")
    tps_files = sorted(glob.glob(str(input_dir / "*.h5")))
    # Exclude processed files
    tps_files = [f for f in tps_files if not f.endswith('_p.h5')]
    
    all_tps = []
    for filepath in tps_files:
        try:
            tps_data = load_tps_data(filepath, tps_indices)
            all_tps.append(tps_data)
        except Exception as e:
            print(f"    WARNING: Could not load TPS from {Path(filepath).name}: {e}")
    
    if all_tps:
        combined_tps = np.concatenate(all_tps)
        print(f"    Loaded {len(combined_tps):,} records from TPS data")
    else:
        print("    WARNING: No TPS data loaded")
        combined_tps = np.full((len(combined_timestamps), len(tps_indices)), np.nan)
    
    # Handle length mismatches between peak and TPS data
    min_rows = min(len(combined_timestamps), len(combined_tps))
    combined_timestamps = combined_timestamps[:min_rows]
    combined_counts = combined_counts[:min_rows]
    combined_tps = combined_tps[:min_rows]
    
    # Adjust peak column count if needed
    n_peak_cols = combined_counts.shape[1]
    if n_peak_cols != len(peak_colnames):
        print(f"    WARNING: Peak data has {n_peak_cols} columns, expected {len(peak_colnames)}")
        if n_peak_cols < len(peak_colnames):
            peak_colnames = peak_colnames[:n_peak_cols]
        else:
            # Extend with generic names
            peak_colnames = peak_colnames + [f"Peak_{i}" for i in range(len(peak_colnames), n_peak_cols)]
    
    # Build DataFrame
    print("\n→ Building output DataFrame...")
    
    # Create column names
    all_colnames = ['date'] + peak_colnames + tps_colnames
    
    # Combine all data
    df = pd.DataFrame(
        np.column_stack([combined_timestamps, combined_counts, combined_tps]),
        columns=all_colnames
    )
    
    # Convert numeric columns
    numeric_cols = peak_colnames + tps_colnames
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"    Created DataFrame with {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def write_cps_csv(
    df: pd.DataFrame,
    date_str: str,
    paths: VOCUSPathConfig
) -> str:
    """
    Write processed VOCUS data to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed VOCUS data
    date_str : str
        Date string in YYYYMMDD format
    paths : VOCUSPathConfig
        Configuration with output paths
    
    Returns
    -------
    str
        Path to output file
    """
    output_path = Path(paths.output_base) / f"{date_str}_cps.parquet"
    
    df.to_parquet(output_path, index=False)
    print(f"  ✓ Wrote {output_path}")
    
    return str(output_path)


# =============================================================================
# Part 3: Integration with Instrument Pipeline
# =============================================================================

def vocus_cps_to_long_format(
    df: pd.DataFrame,
    timezone: str = "America/Los_Angeles"
) -> pd.DataFrame:
    """
    Convert VOCUS cps data to long format for instrument pipeline integration.
    
    This matches the format expected by read_vocus_data() in instrument_pipeline.py.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format VOCUS cps data from process_day()
    timezone : str
        Local timezone for timestamp conversion
    
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        [sample_time, instrument, parameter, value, string]
    """
    print("  → Converting to long format for pipeline integration...")
    
    # Parse timestamps
    df['sample_time'] = pd.to_datetime(df['date'])
    
    # Convert to UTC
    df['sample_time'] = (
        df['sample_time']
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
    
    # Reorder columns to match VanDAQ format
    df_long = df_long[['sample_time', 'instrument', 'parameter', 'value', 'string']]
    
    print(f"    Converted to {len(df_long):,} rows in long format")
    
    return df_long


# =============================================================================
# Main Pipeline
# =============================================================================

def run_vocus_pipeline(
    date_str: str,
    paths: VOCUSPathConfig,
    config: Optional[VOCUSConfig] = None,
    output_long_format: bool = True
) -> Dict[str, Any]:
    """
    Run the complete VOCUS H5 processing pipeline for a single date.
    
    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format
    paths : VOCUSPathConfig
        Input/output paths
    config : VOCUSConfig, optional
        Processing configuration
    output_long_format : bool
        If True, also generate long-format output for pipeline integration
    
    Returns
    -------
    dict
        Dictionary with processing outputs:
        - 'wide': Wide-format DataFrame
        - 'long': Long-format DataFrame (if output_long_format=True)
        - 'csv_path': Path to output CSV
    """
    if config is None:
        config = VOCUSConfig()
    
    outputs = {}
    
    # Process the day
    df = process_day(date_str, paths, config)
    outputs['wide'] = df
    
    if df.empty:
        return outputs
    
    # Write CPS CSV
    csv_path = write_cps_csv(df, date_str, paths)
    outputs['csv_path'] = csv_path
    
    # Generate long format for pipeline integration
    if output_long_format:
        df_long = vocus_cps_to_long_format(df)
        outputs['long'] = df_long
        
        # Write long format CSV
        long_path = Path(paths.output_base) / f"{date_str}_cps_long.parquet"
        df_long.to_parquet(long_path, index=False)
        print(f"  ✓ Wrote long format: {long_path}")
        outputs['long_csv_path'] = str(long_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Completed VOCUS processing for {date_str}")
    print(f"{'='*60}\n")
    
    return outputs


def run_vocus_pipeline_batch(
    date_list: List[str],
    paths: VOCUSPathConfig,
    config: Optional[VOCUSConfig] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run VOCUS pipeline for multiple dates.
    
    Parameters
    ----------
    date_list : list
        List of dates in YYYYMMDD format
    paths : VOCUSPathConfig
        Input/output paths
    config : VOCUSConfig, optional
        Processing configuration
    
    Returns
    -------
    dict
        Dictionary mapping date -> outputs
    """
    all_outputs = {}
    
    for date_str in date_list:
        try:
            outputs = run_vocus_pipeline(date_str, paths, config)
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
        description="Process VOCUS PTR-ToF H5 files to CPS CSV format"
    )
    parser.add_argument(
        'dates',
        nargs='+',
        help='Dates to process (YYYYMMDD format)'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Base input directory containing date folders with H5 files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for processed CSV files'
    )
    parser.add_argument(
        '--peak-columns',
        required=True,
        help='Path to peak columns CSV file'
    )
    parser.add_argument(
        '--tps-columns',
        required=True,
        help='Path to TPS columns CSV file'
    )
    parser.add_argument(
        '--no-long-format',
        action='store_true',
        help='Skip generation of long-format output'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    paths = VOCUSPathConfig(
        input_base=args.input,
        output_base=args.output,
        peak_columns_path=args.peak_columns,
        tps_columns_path=args.tps_columns
    )
    
    # Run pipeline
    results = run_vocus_pipeline_batch(
        args.dates,
        paths,
        config=VOCUSConfig()
    )
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    for date_str, outputs in results.items():
        if 'error' in outputs:
            print(f"  {date_str}: FAILED - {outputs['error']}")
        else:
            n_rows = len(outputs.get('wide', []))
            print(f"  {date_str}: {n_rows:,} records processed")