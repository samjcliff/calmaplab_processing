"""
CalMAPLab Unified Data Processing Pipeline
==========================================

Unified pipeline orchestrating all CalMAPLab mobile measurement data processing:

1. GPS Pipeline: Raw GPS → Kalman smoothed → Road matched → GeoParquet
2. VOCUS H5 Pipeline: Raw .h5 → CPS Parquet (counts per second)
3. VOCUS Calibration Pipeline: CPS → Calibration statistics + zeros
4. Instrument Pipeline: VanDAQ + VOCUS + GPS → Calibrated L2 output

Data Flow:
    ┌─────────────┐     ┌──────────────┐
    │ Raw GPS     │────▶│ GPS Pipeline │────┐
    │ (VanDAQ)    │     │              │    │
    └─────────────┘     └──────────────┘    │
                                            │    ┌──────────────────┐
    ┌─────────────┐     ┌──────────────┐    ├───▶│                  │
    │ VOCUS .h5   │────▶│ VOCUS H5     │────┤    │ Instrument       │────▶ Complete Output
    │ files       │     │ Processor    │    │    │ Pipeline         │
    └─────────────┘     └──────────────┘    │    │                  │
                              │             │    └──────────────────┘
                              ▼             │              ▲
                        ┌──────────────┐    │              │
                        │ VOCUS        │────┘              │
                        │ Calibration  │───────────────────┘
                        └──────────────┘    (cal stats)

Author: S. J. Cliff
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, date
import warnings
import glob
import re
from enum import Enum, auto


# =============================================================================
# Configuration Classes
# =============================================================================

class ProcessingStage(Enum):
    """Pipeline processing stages."""
    GPS = auto()
    VOCUS_H5 = auto()
    VOCUS_CAL = auto()
    INSTRUMENT = auto()
    ALL = auto()


@dataclass
class UnifiedPaths:
    """
    Unified path configuration for all pipeline stages.
    
    Organized by data type and processing stage for clarity.
    """
    # Base directories
    base_raw: str = ""           # Base path for all raw data
    base_processed: str = ""     # Base path for all processed data
    base_reference: str = ""     # Base path for reference files
    
    # GPS paths
    gps_raw: str = ""            # Raw VanDAQ GPS files
    gps_processed: str = ""      # Processed GPS output
    roads_cache: str = ""        # Road segment cache directory
    
    # VOCUS paths
    vocus_h5_raw: str = ""       # Raw VOCUS .h5 files
    vocus_cps: str = ""          # Processed CPS files
    vocus_cal_stats: str = ""    # Calibration statistics output
    vocus_diagnostics: str = ""  # Diagnostic plots
    
    # VanDAQ paths
    vandaq_raw: str = ""         # Raw VanDAQ instrument files
    
    # Calibration reference files
    cal_standards: str = ""      # Calibration cylinder concentrations
    ksens_standards: str = ""    # k_PTR sensitivity standards
    NO_NO2_NOx_O3_cal: str = ""  # NOx/O3 calibration files
    CO_N2O_CH4_C2H6_CO2_cal: str = ""  # Aeris/LICOR calibration files
    
    # Reference files
    peak_columns: str = ""       # VOCUS peak column names
    tps_columns: str = ""        # VOCUS TPS column names
    flags_file: str = ""         # QC flag thresholds
    lag_times: str = ""          # Instrument lag times
    aclima_fields: str = ""      # Aclima field mappings
    exclusion_poly: str = ""     # GeoJSON for spatial filtering
    
    # Output paths
    output_l2: str = ""          # L2 output directory
    output_aclima: str = ""      # Aclima format output
    output_targets: str = ""     # Target calibration output
    
    def __post_init__(self):
        """Auto-populate paths from base directories if not specified."""
        if self.base_raw and not self.gps_raw:
            self.gps_raw = str(Path(self.base_raw) / "vandaq")
        if self.base_raw and not self.vocus_h5_raw:
            self.vocus_h5_raw = str(Path(self.base_raw) / "vocus")
        if self.base_raw and not self.vandaq_raw:
            self.vandaq_raw = str(Path(self.base_raw) / "vandaq")
        
        if self.base_processed and not self.gps_processed:
            self.gps_processed = str(Path(self.base_processed) / "gps")
        if self.base_processed and not self.vocus_cps:
            self.vocus_cps = str(Path(self.base_processed) / "vocus" / "cps")
        if self.base_processed and not self.vocus_cal_stats:
            self.vocus_cal_stats = str(Path(self.base_processed) / "vocus" / "cal_stats")
        if self.base_processed and not self.vocus_diagnostics:
            self.vocus_diagnostics = str(Path(self.base_processed) / "vocus" / "diagnostics")
        if self.base_processed and not self.roads_cache:
            self.roads_cache = str(Path(self.base_processed) / "road_segments")
        if self.base_processed and not self.output_l2:
            self.output_l2 = str(Path(self.base_processed) / "l2")
        if self.base_processed and not self.output_aclima:
            self.output_aclima = str(Path(self.base_processed) / "aclima")
        if self.base_processed and not self.output_targets:
            self.output_targets = str(Path(self.base_processed) / "targets")
    
    def validate(self, stages: List[ProcessingStage] = None) -> List[str]:
        """
        Validate that required paths exist for specified stages.
        
        Returns list of missing/invalid paths.
        """
        missing = []
        
        if stages is None or ProcessingStage.ALL in stages:
            stages = [ProcessingStage.GPS, ProcessingStage.VOCUS_H5, 
                     ProcessingStage.VOCUS_CAL, ProcessingStage.INSTRUMENT]
        
        # GPS stage requirements
        if ProcessingStage.GPS in stages:
            if not self.gps_raw or not Path(self.gps_raw).exists():
                missing.append(f"gps_raw: {self.gps_raw}")
        
        # VOCUS H5 stage requirements
        if ProcessingStage.VOCUS_H5 in stages:
            if not self.vocus_h5_raw or not Path(self.vocus_h5_raw).exists():
                missing.append(f"vocus_h5_raw: {self.vocus_h5_raw}")
            if not self.peak_columns or not Path(self.peak_columns).exists():
                missing.append(f"peak_columns: {self.peak_columns}")
            if not self.tps_columns or not Path(self.tps_columns).exists():
                missing.append(f"tps_columns: {self.tps_columns}")
        
        # VOCUS calibration stage requirements
        if ProcessingStage.VOCUS_CAL in stages:
            if not self.vocus_cps:
                missing.append("vocus_cps: (not set)")
            if not self.cal_standards or not Path(self.cal_standards).exists():
                missing.append(f"cal_standards: {self.cal_standards}")
        
        # Instrument stage requirements
        if ProcessingStage.INSTRUMENT in stages:
            if not self.vandaq_raw or not Path(self.vandaq_raw).exists():
                missing.append(f"vandaq_raw: {self.vandaq_raw}")
            if not self.gps_processed:
                missing.append("gps_processed: (not set)")
            if not self.flags_file or not Path(self.flags_file).exists():
                missing.append(f"flags_file: {self.flags_file}")
            if not self.lag_times or not Path(self.lag_times).exists():
                missing.append(f"lag_times: {self.lag_times}")
            if not self.aclima_fields or not Path(self.aclima_fields).exists():
                missing.append(f"aclima_fields: {self.aclima_fields}")
        
        return missing
    
    def ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        for attr in ['gps_processed', 'vocus_cps', 'vocus_cal_stats', 
                    'vocus_diagnostics', 'roads_cache', 'output_l2',
                    'output_aclima', 'output_targets']:
            path = getattr(self, attr, None)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class UnifiedConfig:
    """
    Unified configuration for all pipeline stages.
    
    Consolidates parameters from all sub-pipelines.
    """
    # Organization info
    org: str = "UCB"
    revision: str = "r1"
    
    # GPS pipeline parameters
    kalman_process_noise: float = 1e-9
    kalman_measurement_noise: float = 1e-10
    kalman_max_displacement_m: float = 15.0
    max_interpolation_gap_s: float = 5.0
    segment_length_m: float = 30.0
    max_match_distance_m: float = 15.0
    max_match_angle_deg: float = 60.0
    
    # VOCUS H5 parameters
    tof_extraction_freq: float = 24990.0
    
    # VOCUS calibration parameters
    cal_cylinder_no: str = ""
    default_k: float = 2.5
    cal_step_flows: List[float] = field(default_factory=lambda: [0, 2, 5, 10])
    flow_tolerance: float = 0.1
    zero_flow_thresh: float = 450.0
    min_cal_samples: int = 70
    cal_window_samples: int = 60
    
    # Instrument pipeline parameters
    ptr_prefixes: str = "HCOKSNV"
    output_types: List[str] = field(default_factory=lambda: ["L1", "L2a"])
    target_cylinders: List[str] = field(default_factory=list)
    target_instruments: List[str] = field(default_factory=list)
    vocus_target_flow: float = 1.0
    target_length: int = 60
    value_precision: int = 5
    coord_precision: int = 5
    
    # Processing options
    verbose: bool = True
    use_cached_roads: bool = True
    generate_diagnostics: bool = True
    
    def __post_init__(self):
        """Generate derived configuration values."""
        self.ptr_regex = f"^[{self.ptr_prefixes}]"


# =============================================================================
# Pipeline Stage Wrappers
# =============================================================================

def run_gps_stage(
    date_str: str,
    paths: UnifiedPaths,
    config: UnifiedConfig
) -> Dict[str, Any]:
    """
    Run GPS processing stage.
    
    Imports and runs gps_pipeline.run_pipeline().
    
    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    paths : UnifiedPaths
        Path configuration
    config : UnifiedConfig
        Processing configuration
    
    Returns
    -------
    dict
        Stage outputs including 'gps_df' and 'segments_df'
    """
    from gps_pipeline import run_pipeline, PipelineConfig
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"GPS STAGE: {date_str}")
        print(f"{'='*60}")
    
    # Find input file
    gps_pattern = f"measurements_van1_{date_str}*.csv"
    gps_files = glob.glob(str(Path(paths.gps_raw) / gps_pattern))
    
    if not gps_files:
        raise FileNotFoundError(f"No GPS data found matching {gps_pattern}")
    
    gps_filepath = gps_files[0]
    
    # Configure GPS pipeline
    gps_config = PipelineConfig(
        kalman_process_noise=config.kalman_process_noise,
        kalman_measurement_noise=config.kalman_measurement_noise,
        kalman_max_displacement_m=config.kalman_max_displacement_m,
        max_interpolation_gap_s=config.max_interpolation_gap_s,
        segment_length_m=config.segment_length_m,
        max_match_distance_m=config.max_match_distance_m,
        max_match_angle_deg=config.max_match_angle_deg,
        roads_filepath=paths.roads_cache
    )
    
    # Output path
    output_filepath = str(Path(paths.gps_processed) / f"processed_gps_{date_str}.parquet")
    
    # Run pipeline
    gps_df, segments_df = run_pipeline(
        gps_filepath,
        output_filepath=output_filepath,
        config=gps_config,
        use_cached_segments=config.use_cached_roads,
        verbose=config.verbose
    )
    
    return {
        'gps_df': gps_df,
        'segments_df': segments_df,
        'output_path': output_filepath
    }


def run_vocus_h5_stage(
    date_str: str,
    paths: UnifiedPaths,
    config: UnifiedConfig
) -> Dict[str, Any]:
    """
    Run VOCUS H5 processing stage.
    
    Imports and runs vocus_h5_processor.run_vocus_pipeline().
    
    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format
    paths : UnifiedPaths
        Path configuration
    config : UnifiedConfig
        Processing configuration
    
    Returns
    -------
    dict
        Stage outputs including 'wide_df' and 'csv_path'
    """
    from vocus_h5_processor import run_vocus_pipeline, VOCUSConfig, VOCUSPathConfig
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"VOCUS H5 STAGE: {date_str}")
        print(f"{'='*60}")
    
    # Configure VOCUS H5 pipeline
    vocus_config = VOCUSConfig(
        tof_extraction_freq=config.tof_extraction_freq
    )
    
    vocus_paths = VOCUSPathConfig(
        input_base=paths.vocus_h5_raw,
        output_base=paths.vocus_cps,
        peak_columns_path=paths.peak_columns,
        tps_columns_path=paths.tps_columns
    )
    
    # Run pipeline
    outputs = run_vocus_pipeline(
        date_str,
        vocus_paths,
        vocus_config,
        output_long_format=False
    )
    
    return outputs


def run_vocus_cal_stage(
    date_str: str,
    paths: UnifiedPaths,
    config: UnifiedConfig
) -> Dict[str, Any]:
    """
    Run VOCUS calibration processing stage.
    
    Imports and runs vocus_calibration.process_vocus_calibrations().
    
    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format
    paths : UnifiedPaths
        Path configuration
    config : UnifiedConfig
        Processing configuration
    
    Returns
    -------
    dict
        Stage outputs including 'cal_stats' and 'zeros'
    """
    from vocus_calibration import (
        process_vocus_calibrations, VOCUSCalConfig, VOCUSCalPathConfig
    )
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"VOCUS CALIBRATION STAGE: {date_str}")
        print(f"{'='*60}")
    
    # Configure VOCUS calibration pipeline
    cal_config = VOCUSCalConfig(
        default_k=config.default_k,
        cal_step_flows=config.cal_step_flows,
        flow_tolerance=config.flow_tolerance,
        zero_flow_thresh=config.zero_flow_thresh,
        min_samples=config.min_cal_samples,
        window_samples=config.cal_window_samples,
        cal_cylinder_no=config.cal_cylinder_no
    )
    
    cal_paths = VOCUSCalPathConfig(
        input_path=paths.vocus_cps,
        output_path=paths.vocus_cps,
        diagnostic_plots_path=paths.vocus_diagnostics,
        cal_stats_path=paths.vocus_cal_stats,
        cal_standard_path=paths.cal_standards,
        ksens_standards_path=paths.ksens_standards
    )
    
    # Run pipeline
    outputs = process_vocus_calibrations(date_str, cal_config, cal_paths)
    
    return outputs


def run_instrument_stage(
    date_str: str,
    paths: UnifiedPaths,
    config: UnifiedConfig
) -> Dict[str, Any]:
    """
    Run instrument processing stage.
    
    Imports and runs instrument_pipeline.run_pipeline().
    
    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format
    paths : UnifiedPaths
        Path configuration
    config : UnifiedConfig
        Processing configuration
    
    Returns
    -------
    dict
        Stage outputs including 'ucb_complete', 'aclima_l2', 'targets'
    """
    from instrument_pipeline import run_pipeline, PipelineConfig, PathConfig
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"INSTRUMENT STAGE: {date_str}")
        print(f"{'='*60}")
    
    # Configure instrument pipeline
    inst_config = PipelineConfig(
        org=config.org,
        revision=config.revision,
        output_types=config.output_types,
        target_cylinders=config.target_cylinders,
        target_instruments=config.target_instruments,
        vocus_target_flow=config.vocus_target_flow,
        target_length=config.target_length,
        ptr_prefixes=config.ptr_prefixes,
        value_precision=config.value_precision,
        coord_precision=config.coord_precision
    )
    
    inst_paths = PathConfig(
        drive_in=paths.vandaq_raw,
        vocus_in=paths.vocus_cps,
        gps_in=paths.gps_processed,
        NO_NO2_NOx_O3_cal_files=paths.NO_NO2_NOx_O3_cal,
        CO_N2O_CH4_C2H6_CO2_cal_files=paths.CO_N2O_CH4_C2H6_CO2_cal,
        vocus_cal_stats=paths.vocus_cal_stats,
        poly=paths.exclusion_poly,
        cylinders=paths.cal_standards,
        flags_file=paths.flags_file,
        lag_times=paths.lag_times,
        aclima_field=paths.aclima_fields,
        aclima_out=paths.output_aclima,
        drive_out=paths.output_l2,
        target_output=paths.output_targets
    )
    
    # Run pipeline
    outputs = run_pipeline(date_str, inst_config, inst_paths)
    
    return outputs


# =============================================================================
# Unified Pipeline Orchestrator
# =============================================================================

class CalMAPLabPipeline:
    """
    Unified pipeline orchestrator for CalMAPLab data processing.
    
    Coordinates all processing stages and manages data flow between them.
    
    Example
    -------
    >>> paths = UnifiedPaths(
    ...     base_raw="/data/raw",
    ...     base_processed="/data/processed",
    ...     cal_standards="/ref/cal_standards.csv",
    ...     ...
    ... )
    >>> config = UnifiedConfig(cal_cylinder_no="CC524064")
    >>> 
    >>> pipeline = CalMAPLabPipeline(paths, config)
    >>> results = pipeline.run("2025-07-15")
    """
    
    def __init__(self, paths: UnifiedPaths, config: UnifiedConfig):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        paths : UnifiedPaths
            Path configuration for all stages
        config : UnifiedConfig
            Processing configuration for all stages
        """
        self.paths = paths
        self.config = config
        self._results = {}
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "CalMAPLabPipeline":
        """
        Create a pipeline from a YAML configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        
        Returns
        -------
        CalMAPLabPipeline
            Configured pipeline instance
        
        Example
        -------
        >>> pipeline = CalMAPLabPipeline.from_yaml("config.yaml")
        >>> results = pipeline.run("2025-07-15")
        """
        import yaml
        
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        paths = UnifiedPaths(**cfg.get("paths", {}))
        config = UnifiedConfig(**cfg.get("config", {}))
        
        return cls(paths, config)
    
    def validate(self, stages: List[ProcessingStage] = None) -> bool:
        """
        Validate configuration before running.
        
        Parameters
        ----------
        stages : list, optional
            Stages to validate. Default: all stages.
        
        Returns
        -------
        bool
            True if valid, raises ValueError otherwise
        """
        missing = self.paths.validate(stages)
        
        if missing:
            raise ValueError(
                f"Missing or invalid paths:\n" +
                "\n".join(f"  - {p}" for p in missing)
            )
        
        return True
    
    def run(
        self,
        date_str: str,
        stages: List[ProcessingStage] = None,
        skip_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Run the unified pipeline for a single date.
        
        Parameters
        ----------
        date_str : str
            Date to process. Accepts YYYY-MM-DD or YYYYMMDD format.
        stages : list, optional
            Specific stages to run. Default: all stages.
        skip_existing : bool
            Skip stages where output already exists
        
        Returns
        -------
        dict
            Results from all stages
        """
        # Normalize date format
        if '-' in date_str:
            date_hyphen = date_str
            date_compact = date_str.replace('-', '')
        else:
            date_compact = date_str
            date_hyphen = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        if stages is None or ProcessingStage.ALL in stages:
            stages = [
                ProcessingStage.GPS,
                ProcessingStage.VOCUS_H5,
                ProcessingStage.VOCUS_CAL,
                ProcessingStage.INSTRUMENT
            ]
        
        # Ensure output directories exist
        self.paths.ensure_output_dirs()
        
        results = {
            'date': date_hyphen,
            'date_compact': date_compact,
            'stages': {}
        }
        
        # Stage 1: GPS
        if ProcessingStage.GPS in stages:
            gps_output = Path(self.paths.gps_processed) / f"processed_gps_{date_hyphen}.parquet"
            
            if skip_existing and gps_output.exists():
                if self.config.verbose:
                    print(f"\n→ Skipping GPS stage (output exists)")
                results['stages']['gps'] = {'skipped': True, 'output_path': str(gps_output)}
            else:
                try:
                    gps_results = run_gps_stage(date_hyphen, self.paths, self.config)
                    results['stages']['gps'] = gps_results
                except Exception as e:
                    results['stages']['gps'] = {'error': str(e)}
                    if self.config.verbose:
                        print(f"  ERROR in GPS stage: {e}")
        
        # Stage 2: VOCUS H5
        if ProcessingStage.VOCUS_H5 in stages:
            vocus_output = Path(self.paths.vocus_cps) / f"{date_compact}_cps.csv"
            
            if skip_existing and vocus_output.exists():
                if self.config.verbose:
                    print(f"\n→ Skipping VOCUS H5 stage (output exists)")
                results['stages']['vocus_h5'] = {'skipped': True, 'csv_path': str(vocus_output)}
            else:
                try:
                    vocus_results = run_vocus_h5_stage(date_compact, self.paths, self.config)
                    results['stages']['vocus_h5'] = vocus_results
                except Exception as e:
                    results['stages']['vocus_h5'] = {'error': str(e)}
                    if self.config.verbose:
                        print(f"  ERROR in VOCUS H5 stage: {e}")
        
        # Stage 3: VOCUS Calibration
        if ProcessingStage.VOCUS_CAL in stages:
            cal_output = Path(self.paths.vocus_cal_stats) / f"{date_compact}_calstats.csv"
            
            if skip_existing and cal_output.exists():
                if self.config.verbose:
                    print(f"\n→ Skipping VOCUS calibration stage (output exists)")
                results['stages']['vocus_cal'] = {'skipped': True}
            else:
                try:
                    cal_results = run_vocus_cal_stage(date_compact, self.paths, self.config)
                    results['stages']['vocus_cal'] = cal_results
                except Exception as e:
                    results['stages']['vocus_cal'] = {'error': str(e)}
                    if self.config.verbose:
                        print(f"  ERROR in VOCUS calibration stage: {e}")
        
        # Stage 4: Instrument
        if ProcessingStage.INSTRUMENT in stages:
            l2_output = Path(self.paths.output_l2) / f"{self.config.org}_complete_{date_compact}_L2a_{self.config.revision}.csv"
            
            if skip_existing and l2_output.exists():
                if self.config.verbose:
                    print(f"\n→ Skipping instrument stage (output exists)")
                results['stages']['instrument'] = {'skipped': True}
            else:
                try:
                    inst_results = run_instrument_stage(date_hyphen, self.paths, self.config)
                    results['stages']['instrument'] = inst_results
                except Exception as e:
                    results['stages']['instrument'] = {'error': str(e)}
                    if self.config.verbose:
                        print(f"  ERROR in instrument stage: {e}")
        
        self._results[date_hyphen] = results
        return results
    
    def run_batch(
        self,
        date_list: List[str],
        stages: List[ProcessingStage] = None,
        skip_existing: bool = False,
        stop_on_error: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the pipeline for multiple dates.
        
        Parameters
        ----------
        date_list : list
            List of dates to process
        stages : list, optional
            Specific stages to run
        skip_existing : bool
            Skip stages where output exists
        stop_on_error : bool
            Stop processing if any date fails
        
        Returns
        -------
        dict
            Results for all dates
        """
        all_results = {}
        
        for date_str in date_list:
            if self.config.verbose:
                print(f"\n{'#'*60}")
                print(f"# Processing: {date_str}")
                print(f"{'#'*60}")
            
            try:
                results = self.run(date_str, stages, skip_existing)
                all_results[date_str] = results
                
                # Check for errors
                errors = [
                    stage for stage, data in results['stages'].items()
                    if isinstance(data, dict) and 'error' in data
                ]
                
                if errors and stop_on_error:
                    print(f"Stopping due to errors in: {errors}")
                    break
                    
            except Exception as e:
                all_results[date_str] = {'error': str(e)}
                if stop_on_error:
                    print(f"Stopping due to error: {e}")
                    break
        
        return all_results
    
    def summary(self) -> pd.DataFrame:
        """
        Generate summary of processed dates.
        
        Returns
        -------
        pd.DataFrame
            Summary with status of each stage per date
        """
        rows = []
        
        for date_str, results in self._results.items():
            row = {'date': date_str}
            
            for stage in ['gps', 'vocus_h5', 'vocus_cal', 'instrument']:
                stage_data = results.get('stages', {}).get(stage, {})
                
                if 'error' in stage_data:
                    row[stage] = 'ERROR'
                elif stage_data.get('skipped'):
                    row[stage] = 'SKIPPED'
                elif stage_data:
                    row[stage] = 'OK'
                else:
                    row[stage] = '-'
            
            rows.append(row)
        
        return pd.DataFrame(rows)


# =============================================================================
# Convenience Functions
# =============================================================================

def process_date(
    date_str: str,
    paths: UnifiedPaths,
    config: UnifiedConfig = None,
    stages: List[ProcessingStage] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a single date.
    
    Parameters
    ----------
    date_str : str
        Date to process
    paths : UnifiedPaths
        Path configuration
    config : UnifiedConfig, optional
        Processing configuration
    stages : list, optional
        Stages to run
    
    Returns
    -------
    dict
        Processing results
    """
    if config is None:
        config = UnifiedConfig()
    
    pipeline = CalMAPLabPipeline(paths, config)
    return pipeline.run(date_str, stages)


def process_date_range(
    start_date: str,
    end_date: str,
    paths: UnifiedPaths,
    config: UnifiedConfig = None,
    stages: List[ProcessingStage] = None,
    skip_existing: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Process a range of dates.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    paths : UnifiedPaths
        Path configuration
    config : UnifiedConfig, optional
        Processing configuration
    stages : list, optional
        Stages to run
    skip_existing : bool
        Skip dates where output exists
    
    Returns
    -------
    dict
        Results for all dates
    """
    if config is None:
        config = UnifiedConfig()
    
    # Generate date list
    dates = pd.date_range(start_date, end_date, freq='D')
    date_list = [d.strftime('%Y-%m-%d') for d in dates]
    
    pipeline = CalMAPLabPipeline(paths, config)
    return pipeline.run_batch(date_list, stages, skip_existing)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for the unified pipeline."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="CalMAPLab Unified Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single date with config file
  python calmaplab_pipeline.py 2025-07-15 --config config.yaml
  
  # Process date range
  python calmaplab_pipeline.py 2025-07-01 2025-07-31 --config config.yaml
  
  # Run only GPS and VOCUS stages
  python calmaplab_pipeline.py 2025-07-15 --config config.yaml --stages gps vocus_h5
  
  # Skip existing outputs
  python calmaplab_pipeline.py 2025-07-15 --config config.yaml --skip-existing
        """
    )
    
    parser.add_argument(
        'dates',
        nargs='+',
        help='Date(s) to process (YYYY-MM-DD). If two dates, treated as range.'
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--stages', '-s',
        nargs='+',
        choices=['gps', 'vocus_h5', 'vocus_cal', 'instrument', 'all'],
        default=['all'],
        help='Processing stages to run'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip stages where output already exists'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop processing if any date fails'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Build paths from YAML
    paths = UnifiedPaths(**yaml_config.get('paths', {}))
    
    # Build config from YAML
    config_dict = yaml_config.get('config', {})
    config_dict['verbose'] = not args.quiet
    config = UnifiedConfig(**config_dict)
    
    # Map stage names to enum
    stage_map = {
        'gps': ProcessingStage.GPS,
        'vocus_h5': ProcessingStage.VOCUS_H5,
        'vocus_cal': ProcessingStage.VOCUS_CAL,
        'instrument': ProcessingStage.INSTRUMENT,
        'all': ProcessingStage.ALL
    }
    stages = [stage_map[s] for s in args.stages]
    
    # Create pipeline
    pipeline = CalMAPLabPipeline(paths, config)
    
    # Validate
    try:
        pipeline.validate(stages)
    except ValueError as e:
        print(f"Configuration error:\n{e}")
        return 1
    
    # Determine date list
    if len(args.dates) == 1:
        date_list = args.dates
    elif len(args.dates) == 2:
        # Date range
        dates = pd.date_range(args.dates[0], args.dates[1], freq='D')
        date_list = [d.strftime('%Y-%m-%d') for d in dates]
    else:
        date_list = args.dates
    
    # Run pipeline
    results = pipeline.run_batch(
        date_list,
        stages=stages,
        skip_existing=args.skip_existing,
        stop_on_error=args.stop_on_error
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    summary = pipeline.summary()
    print(summary.to_string(index=False))
    
    # Count errors
    n_errors = sum(
        1 for r in results.values()
        if 'error' in r or any(
            'error' in s for s in r.get('stages', {}).values()
            if isinstance(s, dict)
        )
    )
    
    if n_errors > 0:
        print(f"\n⚠ {n_errors} date(s) had errors")
        return 1
    
    print(f"\n✓ All {len(date_list)} date(s) processed successfully")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())