# CalMAPLab Unified Data Processing Pipeline

A unified orchestration layer for processing mobile air quality monitoring data from the CalMAPLab platform.

## Overview

This pipeline integrates four processing stages into a single coordinated workflow:

```
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
```

### Pipeline Stages

1. **GPS Pipeline** (`gps_pipeline.py`)
   - Loads raw GPS data from VanDAQ format
   - Applies Kalman smoothing with RTS smoother
   - Downloads/caches road network from OpenStreetMap
   - Matches GPS points to road segments (direction-aware)
   - Calculates drive passes
   - Outputs GeoParquet with segment geometries

2. **VOCUS H5 Pipeline** (`vocus_h5_processor.py`)
   - Loads raw VOCUS PTR-ToF .h5 files
   - Converts counts to counts-per-second (CPS)
   - Extracts timestamps and valve states
   - Outputs processed CSV files

3. **VOCUS Calibration Pipeline** (`vocus_calibration.py`)
   - Identifies calibration windows
   - Computes Deming regression for each species
   - Estimates k_PTR sensitivity relationships
   - Processes in-drive zero measurements
   - Generates diagnostic plots

4. **Instrument Pipeline** (`instrument_pipeline.py`)
   - Loads VanDAQ and VOCUS data
   - Applies instrument-specific lag corrections
   - Flags data based on QC thresholds
   - Joins with processed GPS data
   - Applies calibrations (slope/intercept interpolation)
   - Generates Aclima-format L1/L2 output files

## Installation

### Dependencies

```bash
pip install pandas numpy scipy geopandas shapely h5py pyarrow numba requests pyyaml
```

Optional dependencies:
- `matplotlib` - for diagnostic plots
- `pyreadr` or `rpy2` - for reading R data files

### File Structure

Place all four pipeline modules in your Python path:
```
project/
├── calmaplab_pipeline.py    # Unified orchestrator
├── gps_pipeline.py          # GPS processing
├── vocus_h5_processor.py    # VOCUS H5 processing
├── vocus_calibration.py     # VOCUS calibration
├── instrument_pipeline.py   # Instrument processing
└── config.yaml              # Your configuration
```

## Usage

### Command Line

```bash
# Process a single date
python calmaplab_pipeline.py 2025-07-15 --config config.yaml

# Process a date range
python calmaplab_pipeline.py 2025-07-01 2025-07-31 --config config.yaml

# Run specific stages only
python calmaplab_pipeline.py 2025-07-15 --config config.yaml --stages gps vocus_h5

# Skip existing outputs (incremental processing)
python calmaplab_pipeline.py 2025-07-01 2025-07-31 --config config.yaml --skip-existing

# Quiet mode
python calmaplab_pipeline.py 2025-07-15 --config config.yaml --quiet
```

### Python API

```python
from calmaplab_pipeline import (
    CalMAPLabPipeline, 
    UnifiedPaths, 
    UnifiedConfig,
    ProcessingStage
)

# Configure paths
paths = UnifiedPaths(
    base_raw="/data/calmap/raw",
    base_processed="/data/calmap/processed",
    cal_standards="/ref/calibration_standards.csv",
    peak_columns="/ref/vocus_peak_columns.csv",
    tps_columns="/ref/vocus_tps_columns.csv",
    flags_file="/ref/qc_flags.csv",
    lag_times="/ref/lag_times.csv",
    aclima_fields="/ref/aclima_fields.csv",
)

# Configure processing
config = UnifiedConfig(
    cal_cylinder_no="CC524064",
    org="UCB",
    revision="r1"
)

# Create and run pipeline
pipeline = CalMAPLabPipeline(paths, config)

# Process single date
results = pipeline.run("2025-07-15")

# Process date range
results = pipeline.run_batch(
    ["2025-07-15", "2025-07-16", "2025-07-17"],
    skip_existing=True
)

# Get summary
print(pipeline.summary())
```

### Running Individual Stages

```python
from calmaplab_pipeline import ProcessingStage

# Run only GPS and VOCUS H5 stages
results = pipeline.run(
    "2025-07-15",
    stages=[ProcessingStage.GPS, ProcessingStage.VOCUS_H5]
)

# Run only calibration (assumes VOCUS CPS files exist)
results = pipeline.run(
    "2025-07-15",
    stages=[ProcessingStage.VOCUS_CAL]
)
```

## Configuration

### YAML Configuration File

Create a `config.yaml` file (see `config_example.yaml` for template):

```yaml
paths:
  base_raw: "/data/calmap/raw"
  base_processed: "/data/calmap/processed"
  cal_standards: "/ref/calibration_standards.csv"
  # ... other paths

config:
  org: "UCB"
  revision: "r1"
  cal_cylinder_no: "CC524064"
  # ... other settings
```

### Path Configuration

| Path | Description | Required For |
|------|-------------|--------------|
| `gps_raw` | Raw VanDAQ GPS files | GPS stage |
| `gps_processed` | Processed GPS output | GPS, Instrument |
| `vocus_h5_raw` | Raw VOCUS .h5 files | VOCUS H5 stage |
| `vocus_cps` | Processed CPS files | VOCUS H5, Cal, Instrument |
| `cal_standards` | Calibration cylinder concentrations | VOCUS Cal |
| `peak_columns` | VOCUS peak column names CSV | VOCUS H5 |
| `tps_columns` | VOCUS TPS column names CSV | VOCUS H5 |
| `flags_file` | QC flag thresholds | Instrument |
| `lag_times` | Instrument lag times | Instrument |
| `aclima_fields` | Field name mappings | Instrument |

### Processing Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kalman_process_noise` | 1e-9 | Kalman filter process noise |
| `kalman_measurement_noise` | 1e-10 | Kalman filter measurement noise |
| `segment_length_m` | 30 | Road segment length (meters) |
| `max_match_distance_m` | 15 | Max GPS-to-road matching distance |
| `cal_cylinder_no` | - | Active calibration cylinder ID |
| `default_k` | 2.5 | Default k_PTR value |
| `ptr_prefixes` | "HCOKSNV" | Regex prefixes for PTR ions |

## Output Files

### GPS Stage
- `processed_gps_{date}.parquet` - GeoParquet with road-matched GPS

### VOCUS H5 Stage
- `{date}_cps.csv` - Wide-format CPS data
- `{date}_cps_long.csv` - Long-format for pipeline integration

### VOCUS Calibration Stage
- `{date}_calstats.csv` - Calibration statistics (slopes, intercepts)
- `{date}_zeros.csv` - In-drive zero measurements
- `{date}_curves.pdf` - Diagnostic calibration curves
- `{date}_ksens.pdf` - k_PTR sensitivity plot

### Instrument Stage
- `UCB_complete_{date}_L2a_r1.csv` - Full 1Hz calibrated data
- `UCB_{date}_L2a_r1.csv` - Aclima-format L2 output
- `{date}_targets.csv` - Target calibration results

## Error Handling

The pipeline handles errors gracefully:

```python
results = pipeline.run_batch(date_list, stop_on_error=False)

# Check for errors
for date, result in results.items():
    if 'error' in result:
        print(f"{date}: {result['error']}")
    for stage, data in result.get('stages', {}).items():
        if isinstance(data, dict) and 'error' in data:
            print(f"{date} {stage}: {data['error']}")
```

## Extending the Pipeline

### Adding Custom Stages

```python
from calmaplab_pipeline import CalMAPLabPipeline, ProcessingStage

class ExtendedPipeline(CalMAPLabPipeline):
    def run_custom_stage(self, date_str: str):
        # Your custom processing
        pass
    
    def run(self, date_str, stages=None, **kwargs):
        results = super().run(date_str, stages, **kwargs)
        
        # Add custom stage
        if self.should_run_custom_stage:
            results['stages']['custom'] = self.run_custom_stage(date_str)
        
        return results
```

## License
