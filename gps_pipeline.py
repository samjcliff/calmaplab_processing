"""
GPS Processing Pipeline for Mobile Air Quality Monitoring
==========================================================

Complete pipeline for processing GPS data from mobile monitoring platforms:
1. Load and clean GPS data from VanDAQ format
2. Apply Kalman smoothing with RTS smoother
3. Download and segment road network from OpenStreetMap (with grid-based caching)
4. Match GPS points to road segments (direction-aware)
5. Calculate drive passes
6. Export as GeoParquet with segment line geometries

Author: S. J. Cliff and H. M. Byrne
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import inv
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import warnings
import glob
from datetime import datetime, date
import random

# Optional imports - will be checked at runtime
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon, box
    from shapely.ops import unary_union
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    warnings.warn("geopandas not available - GeoParquet export disabled")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests not available - OSM download disabled")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration parameters for the GPS processing pipeline."""
    
    # Kalman filter parameters
    kalman_process_noise: float = 1e-9
    kalman_measurement_noise: float = 1e-10
    kalman_max_displacement_m: float = 15.0
    
    # Gap interpolation
    max_interpolation_gap_s: float = 5.0
    
    # Road segmentation
    segment_length_m: float = 30.0
    min_segment_length_m: float = 15.0
    
    # Road matching
    max_match_distance_m: float = 15.0
    max_match_angle_deg: float = 60.0
    min_speed_for_direction_mph: float = 2.0
    kdtree_radius_m: float = 100.0
    
    # OSM query settings
    osm_highway_types: List[str] = None
    osm_buffer_m: float = 200.0
    grid_size_deg: float = 0.1  # Grid cell size in degrees
    
    # Output settings
    output_crs: str = "EPSG:4326"
    
    # Road segments directory
    roads_filepath: str = "data/processed/road_segments/"
    
    def __post_init__(self):
        if self.osm_highway_types is None:
            self.osm_highway_types = [
                'motorway', 'motorway_link',
                'trunk', 'trunk_link',
                'primary', 'primary_link',
                'secondary', 'secondary_link',
                'tertiary', 'tertiary_link',
                'unclassified', 'residential',
                'living_street', 'service'
            ]


# =============================================================================
# Part 1: GPS Data Loading and Kalman Smoothing
# =============================================================================

def read_gps_data(filepath: str, 
                  gps_parameters: List[str] = None) -> pd.DataFrame:
    """
    Read GPS data from VanDAQ long-format CSV.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    gps_parameters : list, optional
        Parameter names to extract. Default: ['latitude', 'longitude', 'speed', 'direction']
    
    Returns
    -------
    pd.DataFrame
        Wide-format GPS data with columns: datetime, lat, lon, gps_speed_mps, gps_direction_deg
    """
    if gps_parameters is None:
        gps_parameters = ['latitude', 'longitude', 'speed', 'direction']
    
    df = pd.read_csv(filepath)
    
    # Check if already wide format
    if 'latitude' in df.columns or 'lat' in df.columns:
        # Already wide format
        df['datetime'] = pd.to_datetime(df['sample_time'] if 'sample_time' in df.columns else df['datetime'])
        col_map = {
            'latitude': 'lat', 'longitude': 'lon',
            'speed': 'gps_speed_mps', 'direction': 'gps_direction_deg'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        return df
    
    # Long format - pivot
    df['sample_time'] = pd.to_datetime(df['sample_time'])
    df['sample_time_rounded'] = df['sample_time'].dt.round('1s')
    
    # Filter to GPS parameters
    gps_df = df[df['parameter'].isin(gps_parameters)].copy()
    
    # Pivot to wide format
    wide = gps_df.pivot_table(
        index='sample_time_rounded',
        columns='parameter',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    wide = wide.rename(columns={
        'sample_time_rounded': 'datetime',
        'latitude': 'lat',
        'longitude': 'lon',
        'speed': 'gps_speed_mps',
        'direction': 'gps_direction_deg'
    })
    
    return wide.sort_values('datetime').reset_index(drop=True)


def clean_gps_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid coordinates and duplicates."""
    df = df.copy()
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
    df = df.sort_values('datetime')
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    return df.reset_index(drop=True)


class GPSKalmanFilter:
    """
    Kalman filter with RTS smoother for GPS trajectory smoothing.
    
    State vector: [lat, lon, v_lat, v_lon]
    Observation: [lat, lon]
    """
    
    def __init__(self, process_noise: float = 1e-9, measurement_noise: float = 1e-10):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def _F(self, dt: float) -> np.ndarray:
        """State transition matrix."""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def _Q(self, dt: float) -> np.ndarray:
        """Process noise covariance."""
        q = self.process_noise
        return np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ]) * q
    
    def _H(self) -> np.ndarray:
        """Observation matrix."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
    
    def _R(self) -> np.ndarray:
        """Measurement noise covariance."""
        return np.eye(2) * self.measurement_noise
    
    def smooth(self, timestamps: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter with RTS smoother.
        
        Returns
        -------
        lat_smooth, lon_smooth : np.ndarray
            Smoothed coordinates
        """
        n = len(timestamps)
        if n < 2:
            return lat.copy(), lon.copy()
        
        # Convert timestamps to seconds
        if hasattr(timestamps[0], 'timestamp'):
            times = np.array([t.timestamp() for t in timestamps])
        else:
            times = timestamps.astype('int64') / 1e9
        
        H = self._H()
        R = self._R()
        
        # Initialize
        x = np.zeros((n, 4))
        P = np.zeros((n, 4, 4))
        x_pred = np.zeros((n, 4))
        P_pred = np.zeros((n, 4, 4))
        
        x[0] = [lat[0], lon[0], 0, 0]
        P[0] = np.eye(4) * 1e-6
        
        # Forward pass
        for i in range(1, n):
            dt = times[i] - times[i-1]
            if dt <= 0:
                dt = 1.0
            
            F = self._F(dt)
            Q = self._Q(dt)
            
            # Predict
            x_pred[i] = F @ x[i-1]
            P_pred[i] = F @ P[i-1] @ F.T + Q
            
            # Update
            z = np.array([lat[i], lon[i]])
            y = z - H @ x_pred[i]
            S = H @ P_pred[i] @ H.T + R
            K = P_pred[i] @ H.T @ inv(S)
            
            x[i] = x_pred[i] + K @ y
            P[i] = (np.eye(4) - K @ H) @ P_pred[i]
        
        # Backward pass (RTS smoother)
        x_smooth = np.zeros((n, 4))
        x_smooth[-1] = x[-1]
        
        for i in range(n-2, -1, -1):
            dt = times[i+1] - times[i]
            if dt <= 0:
                dt = 1.0
            F = self._F(dt)
            
            G = P[i] @ F.T @ inv(P_pred[i+1])
            x_smooth[i] = x[i] + G @ (x_smooth[i+1] - x_pred[i+1])
        
        return x_smooth[:, 0], x_smooth[:, 1]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in meters."""
    R = 6371000
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def limit_kalman_displacement(lat_raw, lon_raw, lat_kf, lon_kf, max_displacement_m: float = 15.0):
    """Revert to raw coordinates where Kalman displacement exceeds threshold."""
    displacement = haversine_distance(lat_raw, lon_raw, lat_kf, lon_kf)
    exceeded = displacement > max_displacement_m
    lat_out = np.where(exceeded, lat_raw, lat_kf)
    lon_out = np.where(exceeded, lon_raw, lon_kf)
    return lat_out, lon_out


def calculate_speed_bearing(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed and bearing from smoothed coordinates."""
    df = df.copy()
    
    lat = df['lat_kf'].values
    lon = df['lon_kf'].values
    times = df['datetime'].values
    
    n = len(df)
    speed_mps = np.zeros(n)
    bearing_deg = np.zeros(n)
    
    for i in range(1, n):
        dist = haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i])
        dt = (times[i] - times[i-1]) / np.timedelta64(1, 's')
        
        if dt > 0:
            speed_mps[i] = dist / dt
        
        # Bearing
        lat1_r, lat2_r = np.radians(lat[i-1]), np.radians(lat[i])
        dlon = np.radians(lon[i] - lon[i-1])
        x = np.sin(dlon) * np.cos(lat2_r)
        y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
        bearing_deg[i] = (np.degrees(np.arctan2(x, y)) + 360) % 360
    
    # Use GPS-reported values if available, otherwise calculated
    if 'gps_speed_mps' in df.columns:
        df['car_speed_mph'] = np.where(
            df['gps_speed_mps'].notna(),
            df['gps_speed_mps'] * 2.23694,
            speed_mps * 2.23694
        )
    else:
        df['car_speed_mph'] = speed_mps * 2.23694
    
    if 'gps_direction_deg' in df.columns:
        df['car_bearing_deg'] = np.where(
            df['gps_direction_deg'].notna(),
            df['gps_direction_deg'],
            bearing_deg
        )
    else:
        df['car_bearing_deg'] = bearing_deg
    
    return df


def interpolate_to_1hz(df: pd.DataFrame, max_gap_s: float = 5.0) -> pd.DataFrame:
    """
    Interpolate GPS data to complete 1 Hz time series.
    
    - Creates complete second-by-second datetime index
    - Raw lat/lon remain NA where no measurement
    - Smoothed lat_kf/lon_kf are interpolated (up to max_gap_s)
    - Speed/bearing are interpolated
    
    Parameters
    ----------
    df : pd.DataFrame
        GPS data with datetime, lat, lon, lat_kf, lon_kf, car_speed_mph, car_bearing_deg
    max_gap_s : float
        Maximum gap to interpolate (seconds)
    
    Returns
    -------
    pd.DataFrame
        Complete 1 Hz time series
    """
    df = df.copy()
    
    # Ensure datetime is timezone-naive for reindexing (or consistent tz)
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    # Create complete 1 Hz index
    min_time = df['datetime'].min().floor('s')
    max_time = df['datetime'].max().ceil('s')
    complete_index = pd.date_range(min_time, max_time, freq='1s')
    
    # Round original datetimes to nearest second for joining
    df['datetime_rounded'] = df['datetime'].dt.round('s')
    
    # Aggregate duplicates (multiple readings in same second)
    agg_cols = {
        'lat': 'mean',
        'lon': 'mean', 
        'lat_kf': 'mean',
        'lon_kf': 'mean',
        'car_speed_mph': 'mean',
        'car_bearing_deg': 'mean'
    }
    # Only aggregate columns that exist
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
    
    # Also keep other columns (first value)
    other_cols = [c for c in df.columns if c not in agg_cols and c not in ['datetime', 'datetime_rounded']]
    for col in other_cols:
        agg_cols[col] = 'first'
    
    df_agg = df.groupby('datetime_rounded').agg(agg_cols).reset_index()
    df_agg = df_agg.rename(columns={'datetime_rounded': 'datetime'})
    
    # Create complete dataframe
    complete_df = pd.DataFrame({'datetime': complete_index})
    complete_df = complete_df.merge(df_agg, on='datetime', how='left')
    
    # Interpolate smoothed coordinates (not raw)
    # Use linear interpolation with max gap
    for col in ['lat_kf', 'lon_kf', 'car_speed_mph', 'car_bearing_deg']:
        if col in complete_df.columns:
            # Find gaps
            is_null = complete_df[col].isna()
            
            # Create gap groups
            gap_group = (~is_null).cumsum()
            gap_sizes = complete_df.groupby(gap_group)[col].transform('size')
            
            # Only interpolate where gap <= max_gap_s
            # First, do the interpolation
            interpolated = complete_df[col].interpolate(method='linear')
            
            # Then mask out where gaps are too large
            # A gap is "too large" if consecutive NAs exceed max_gap_s
            null_runs = is_null.astype(int).groupby((~is_null).cumsum()).cumsum()
            too_large = null_runs > max_gap_s
            
            complete_df[col] = np.where(too_large, np.nan, interpolated)
    
    # Raw lat/lon stay as NA where no measurement (do not interpolate)
    # They're already NA from the left join
    
    return complete_df


def process_gps_part1(filepath: str, config: PipelineConfig = None) -> pd.DataFrame:
    """
    Part 1: Load, clean, smooth, and interpolate GPS data to 1 Hz.
    
    Returns DataFrame with columns:
    - datetime (complete 1 Hz time series)
    - lat, lon (raw - NA where no measurement)
    - lat_kf, lon_kf (Kalman smoothed, interpolated up to 5s gaps)
    - car_speed_mph, car_bearing_deg (interpolated)
    """
    if config is None:
        config = PipelineConfig()
    
    # Load and clean
    df = read_gps_data(filepath)
    df = clean_gps_data(df)
    
    # Kalman smoothing
    kf = GPSKalmanFilter(
        process_noise=config.kalman_process_noise,
        measurement_noise=config.kalman_measurement_noise
    )
    lat_kf, lon_kf = kf.smooth(df['datetime'].values, df['lat'].values, df['lon'].values)
    
    # Limit displacement
    lat_kf, lon_kf = limit_kalman_displacement(
        df['lat'].values, df['lon'].values,
        lat_kf, lon_kf,
        max_displacement_m=config.kalman_max_displacement_m
    )
    
    df['lat_kf'] = lat_kf
    df['lon_kf'] = lon_kf
    
    # Calculate speed and bearing
    df = calculate_speed_bearing(df)
    
    # Interpolate to complete 1 Hz time series
    df = interpolate_to_1hz(df, max_gap_s=config.max_interpolation_gap_s)
    
    return df


# =============================================================================
# Part 2: Road Network Download and Segmentation (with Grid-Based Caching)
# =============================================================================

# Overpass API servers for rotation
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

CRS_WGS = 4326


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def calculate_linestring_angle(coords: List[Tuple[float, float]]) -> float:
    """Calculate the overall bearing of a linestring from its coordinates."""
    if len(coords) < 2:
        return 0.0
    # Use first and last point for overall direction
    lat1, lon1 = coords[0]
    lat2, lon2 = coords[-1]
    return calculate_bearing(lat1, lon1, lat2, lon2)


def cluster_by_degree(directions: np.ndarray, threshold: float = 90.0) -> np.ndarray:
    """
    Cluster road directions to separate opposing travel paths.
    
    Used to distinguish between, e.g., northbound and southbound lanes
    of the same named road when they're marked as oneway.
    """
    if len(directions) == 0:
        return np.array([])
    
    # Simple clustering: if direction differs by more than threshold, different cluster
    clusters = np.zeros(len(directions), dtype=int)
    
    if len(directions) == 1:
        return clusters
    
    # Get mean direction
    mean_dir = np.mean(directions)
    
    for i, d in enumerate(directions):
        # Calculate angular difference from mean
        diff = abs(d - mean_dir)
        if diff > 180:
            diff = 360 - diff
        
        if diff > threshold / 2:
            clusters[i] = 1
    
    return clusters


def osm_fetch_with_retry(query: str,
                         servers: List[str] = None,
                         retries_per_server: int = 2,
                         base_sleep: float = 1.0,
                         max_sleep: float = 8.0,
                         timeout: int = 120,
                         verbose: bool = True) -> dict:
    """
    Robust Overpass API fetch with retries and server rotation.
    
    Mimics the R osm_fetch_sf function behavior.
    
    Parameters
    ----------
    query : str
        Overpass QL query
    servers : list
        List of Overpass server URLs
    retries_per_server : int
        Number of retry attempts per server
    base_sleep : float
        Base sleep time for exponential backoff
    max_sleep : float
        Maximum sleep time
    timeout : int
        Request timeout in seconds
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        OSM data response
    """

    if verbose:
        print(f"Query being sent:\n{query}")

    if not HAS_REQUESTS:
        raise RuntimeError("requests library required for OSM download")
    
    if servers is None:
        servers = OVERPASS_SERVERS
    
    last_error = None
    
    for server in servers:
        if verbose:
            print(f"  Trying Overpass server: {server}")
        
        for attempt in range(retries_per_server):
            # Jittered exponential backoff after first try
            if attempt > 0:
                pause = min(max_sleep, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                if verbose:
                    print(f"    Retry {attempt + 1} (sleep {pause:.1f}s)...")
                time.sleep(pause)
            
            try:
                response = requests.post(
                    server,
                    data={'data': query},
                    timeout=timeout,
                    headers={'User-Agent': 'CalMAPLab-GPS-Pipeline/1.0'}
                )
                response.raise_for_status()
                data = response.json()
                
                # Basic sanity check: did we get any elements?
                elements = data.get('elements', [])
                if verbose:
                    print(f"    Received {len(elements)} elements")
                
                return data
                
            except Exception as e:
                last_error = e
                if verbose:
                    print(f"    Error: {e}")
    
    raise RuntimeError(f"All Overpass attempts failed. Last error: {last_error}")


def build_overpass_query_for_bbox(bbox: Tuple[float, float, float, float], 
                                   highway_types: List[str]) -> str:
    """
    Build Overpass QL query for roads within a bounding box.
    
    Parameters
    ----------
    bbox : tuple
        (south, west, north, east) bounding box
    highway_types : list
        List of highway types to include
    
    Returns
    -------
    str
        Overpass QL query
    """
    south, west, north, east = bbox
    highway_filter = '|'.join(highway_types)
    bbox_str = f"{south},{west},{north},{east}"
    
    return f"""
[out:json][timeout:120];
(
  way["highway"~"^({highway_filter})$"]({bbox_str});
);
out body;
>;
out skel qt;
"""


def parse_osm_roads(osm_data: dict, highway_types: List[str] = None) -> List[dict]:
    """
    Parse Overpass response into road features.
    
    Parameters
    ----------
    osm_data : dict
        Overpass API response
    highway_types : list, optional
        Highway types to filter (if None, all are included)
    
    Returns
    -------
    list
        List of road dictionaries with coords, tags, etc.
    """
    elements = osm_data.get('elements', [])
    
    # Build node lookup
    nodes = {}
    ways = []
    
    for element in elements:
        if element['type'] == 'node':
            nodes[element['id']] = (element['lat'], element['lon'])
        elif element['type'] == 'way':
            ways.append(element)
    
    roads = []
    for way in ways:
        tags = way.get('tags', {})
        highway = tags.get('highway', 'unknown')
        
        # Filter by highway type if specified
        if highway_types and highway not in highway_types:
            continue
        
        # Build coordinate list
        coords = []
        for node_id in way.get('nodes', []):
            if node_id in nodes:
                coords.append(nodes[node_id])
        
        if len(coords) < 2:
            continue
        
        # Handle oneway
        oneway_tag = tags.get('oneway', 'no')
        if oneway_tag == 'yes':
            oneway = True
        elif oneway_tag == '-1':
            oneway = True
            coords = coords[::-1]  # Reverse direction
        else:
            oneway = False
        
        # Calculate direction
        dir_deg = calculate_linestring_angle(coords)
        
        roads.append({
            'osm_id': way['id'],
            'name': tags.get('name'),
            'highway': highway,
            'oneway': oneway,
            'oneway_tag': oneway_tag,
            'service': tags.get('service'),
            'dir_deg': dir_deg,
            'coords': coords
        })
    
    return roads


def segment_road(coords: List[Tuple[float, float]], 
                 segment_length: float = 30.0,
                 min_length: float = 15.0) -> List[dict]:
    """
    Divide a road into fixed-length segments.
    
    Mimics the R lixelize_lines behavior.
    
    Parameters
    ----------
    coords : list
        List of (lat, lon) tuples
    segment_length : float
        Target segment length in meters
    min_length : float
        Minimum segment length in meters
    
    Returns
    -------
    list
        List of segment dictionaries
    """
    if len(coords) < 2:
        return []
    
    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(coords)):
        d = haversine_distance(coords[i-1][0], coords[i-1][1], 
                               coords[i][0], coords[i][1])
        distances.append(distances[-1] + d)
    
    total_length = distances[-1]
    
    # If road is too short, return single segment
    if total_length < min_length:
        center_idx = len(coords) // 2
        return [{
            'start': coords[0],
            'end': coords[-1],
            'center': coords[center_idx],
            'bearing': calculate_bearing(coords[0][0], coords[0][1], 
                                        coords[-1][0], coords[-1][1]),
            'length': total_length
        }]
    
    # Calculate number of segments (end segments will be 50-150% of target length)
    n_segments = max(1, round(total_length / segment_length))
    actual_length = total_length / n_segments
    
    def point_at_distance(target_dist):
        """Interpolate point at given distance along the line."""
        if target_dist <= 0:
            return coords[0]
        if target_dist >= distances[-1]:
            return coords[-1]
        
        for i in range(1, len(distances)):
            if distances[i] >= target_dist:
                frac = (target_dist - distances[i-1]) / (distances[i] - distances[i-1])
                lat = coords[i-1][0] + frac * (coords[i][0] - coords[i-1][0])
                lon = coords[i-1][1] + frac * (coords[i][1] - coords[i-1][1])
                return (lat, lon)
        
        return coords[-1]
    
    segments = []
    for i in range(n_segments):
        start_pt = point_at_distance(i * actual_length)
        end_pt = point_at_distance((i + 1) * actual_length)
        center_pt = point_at_distance((i + 0.5) * actual_length)
        
        segments.append({
            'start': start_pt,
            'end': end_pt,
            'center': center_pt,
            'bearing': calculate_bearing(start_pt[0], start_pt[1], 
                                        end_pt[0], end_pt[1]),
            'length': actual_length
        })
    
    return segments


def create_road_segments(bbox: Tuple[float, float, float, float] = None,
                         bpoly: 'gpd.GeoDataFrame' = None,
                         length_seg: float = 30,
                         highway_types: List[str] = None,
                         clip_osm: bool = True,
                         verbose: bool = True) -> 'gpd.GeoDataFrame':
    """
    Download OSM roads and divide into segments of a set length.
    
    Mimics the R create_road_segments function.
    
    Parameters
    ----------
    bbox : tuple, optional
        (south, west, north, east) bounding box
    bpoly : GeoDataFrame, optional
        Bounding polygon for clipping
    length_seg : float
        Segment length in meters (default: 30)
    highway_types : list, optional
        Highway types to include
    clip_osm : bool
        Whether to clip roads to bpoly extent
    verbose : bool
        Print progress
    
    Returns
    -------
    GeoDataFrame
        Road segments with geometry and attributes
    """
    if not HAS_GEOPANDAS:
        raise RuntimeError("geopandas required for road segmentation")
    
    if highway_types is None:
        highway_types = [
            'motorway', 'motorway_link',
            'trunk', 'trunk_link',
            'primary', 'primary_link',
            'secondary', 'secondary_link',
            'tertiary', 'tertiary_link',
            'unclassified', 'residential',
            'living_street', 'service'
        ]
    
    # Get bounding box from polygon if needed
    if clip_osm and bpoly is not None:
        bounds = bpoly.total_bounds  # (minx, miny, maxx, maxy)
        bbox = (bounds[1], bounds[0], bounds[3], bounds[2])  # (south, west, north, east)
    
    if bbox is None:
        raise ValueError("Either bbox or bpoly must be provided")
    
    # 1. Download OSM data
    if verbose:
        print(f"  Downloading OSM roads for bbox: {bbox}")
    
    query = build_overpass_query_for_bbox(bbox, highway_types)
    osm_data = osm_fetch_with_retry(query, verbose=verbose)
    
    # 2. Parse roads
    roads = parse_osm_roads(osm_data, highway_types)
    
    if verbose:
        print(f"  Downloaded {len(roads)} roads")
    
    if len(roads) == 0:
        # Return empty GeoDataFrame with correct schema
        return gpd.GeoDataFrame({
            f'lon_{int(length_seg)}': [],
            f'lat_{int(length_seg)}': [],
            'rd_type': [],
            'rd_name': [],
            'oneway': [],
            'service': [],
            'rd_deg': [],
            'geometry': []
        }, crs=f"EPSG:{CRS_WGS}")
    
    # 3. Group roads by name and cluster oneway roads
    roads_df = pd.DataFrame(roads)
    
     # Add cluster for oneway roads (to separate opposing directions)
    roads_df['cluster'] = -1
    
    if 'name' in roads_df.columns and len(roads_df) > 0:
        # Fill NaN names with a placeholder for grouping, then group
        name_filled = roads_df['name'].fillna('__unnamed__')
        
        for name, group_idx in roads_df.groupby(name_filled).groups.items():
            group = roads_df.loc[group_idx]
            if len(group) > 0 and group['oneway'].any() and group['highway'].iloc[0] != 'service':
                roads_df.loc[group_idx, 'cluster'] = cluster_by_degree(group['dir_deg'].values)
    
    # 4. Create LineStrings and optionally clip to bpoly
    geometries = []
    for _, road in roads_df.iterrows():
        coords = road['coords']
        if len(coords) >= 2:
            # Note: shapely uses (lon, lat) order
            line = LineString([(c[1], c[0]) for c in coords])
            geometries.append(line)
        else:
            geometries.append(None)
    
    roads_gdf = gpd.GeoDataFrame(roads_df, geometry=geometries, crs=f"EPSG:{CRS_WGS}")
    roads_gdf = roads_gdf[roads_gdf.geometry.notna()]
    
    if clip_osm and bpoly is not None:
        # Clip to bounding polygon
        roads_gdf = gpd.clip(roads_gdf, bpoly)
    
    # 5. Union continuous road sections by name/oneway/cluster
    # Group and union geometries
    grouped = roads_gdf.groupby(['highway', 'name', 'oneway', 'cluster', 'service'], dropna=False)
    
    merged_roads = []
    for keys, group in grouped:
        if len(group) == 0:
            continue
        
        # Union all geometries in this group
        merged_geom = unary_union(group.geometry)
        
        # Handle MultiLineString by exploding to LineStrings
        if merged_geom.geom_type == 'MultiLineString':
            for geom in merged_geom.geoms:
                merged_roads.append({
                    'highway': keys[0],
                    'name': keys[1],
                    'oneway': keys[2],
                    'cluster': keys[3],
                    'service': keys[4],
                    'geometry': geom
                })
        else:
            merged_roads.append({
                'highway': keys[0],
                'name': keys[1],
                'oneway': keys[2],
                'cluster': keys[3],
                'service': keys[4],
                'geometry': merged_geom
            })
    
    if len(merged_roads) == 0:
        return gpd.GeoDataFrame({
            f'lon_{int(length_seg)}': [],
            f'lat_{int(length_seg)}': [],
            'rd_type': [],
            'rd_name': [],
            'oneway': [],
            'service': [],
            'rd_deg': [],
            'geometry': []
        }, crs=f"EPSG:{CRS_WGS}")
    
    roads_gdf = gpd.GeoDataFrame(merged_roads, crs=f"EPSG:{CRS_WGS}")
    
    # 6. Segment each road
    if verbose:
        print(f"  Splitting roads into {length_seg}m segments...")
    
    all_segments = []
    
    for _, road in roads_gdf.iterrows():
        geom = road['geometry']
        
        if geom is None or geom.is_empty:
            continue
        
        # Convert to (lat, lon) coordinate list
        if geom.geom_type == 'LineString':
            coords = [(y, x) for x, y in geom.coords]
        else:
            continue
        
        # Segment the road
        segments = segment_road(coords, length_seg, min_length=length_seg/2)
        
        for seg in segments:
            # Create LineString geometry for segment (lon, lat order for shapely)
            seg_geom = LineString([
                (seg['start'][1], seg['start'][0]),
                (seg['end'][1], seg['end'][0])
            ])
            
            all_segments.append({
                f'lon_{int(length_seg)}': round(seg['center'][1], 6),
                f'lat_{int(length_seg)}': round(seg['center'][0], 6),
                'rd_type': road['highway'],
                'rd_name': road['name'],
                'oneway': road['oneway'] if road['oneway'] != 'no' else False,
                'service': road['service'],
                'rd_deg': seg['bearing'],
                'geometry': seg_geom
            })
    
    if len(all_segments) == 0:
        return gpd.GeoDataFrame({
            f'lon_{int(length_seg)}': [],
            f'lat_{int(length_seg)}': [],
            'rd_type': [],
            'rd_name': [],
            'oneway': [],
            'service': [],
            'rd_deg': [],
            'geometry': []
        }, crs=f"EPSG:{CRS_WGS}")
    
    segments_gdf = gpd.GeoDataFrame(all_segments, crs=f"EPSG:{CRS_WGS}")
    
    if verbose:
        print(f"  Created {len(segments_gdf)} road segments")
    
    return segments_gdf


def draw_grid_box(x: float, y: float, width: float, center: bool = True) -> List[Tuple[float, float]]:
    """
    Draw a grid box polygon coordinates.
    
    Parameters
    ----------
    x, y : float
        Reference coordinates (center or corner)
    width : float
        Box width in degrees
    center : bool
        If True, x/y is center; if False, x/y is bottom-left corner
    
    Returns
    -------
    list
        List of (x, y) corner coordinates forming closed polygon
    """
    if center:
        x_all = [x + width * dx for dx in [-0.5, -0.5, 0.5, 0.5, -0.5]]
        y_all = [y + width * dy for dy in [-0.5, 0.5, 0.5, -0.5, -0.5]]
    else:
        x_all = [x + width * dx for dx in [0, 0, 1, 1, 0]]
        y_all = [y + width * dy for dy in [0, 1, 1, 0, 0]]
    
    return list(zip(x_all, y_all))


def create_grid_polygon(xmin: float, ymin: float, width: float = 0.1) -> 'Polygon':
    """Create a grid cell polygon."""
    return Polygon([
        (xmin, ymin),
        (xmin, ymin + width),
        (xmin + width, ymin + width),
        (xmin + width, ymin),
        (xmin, ymin)
    ])


class RoadSegmentManager:
    """
    Manages grid-based road segment caching with optimized loading.
    """
    
    SUMMARY_FILENAME = "road_segments_summary.geojson"
    
    def __init__(self,
                 filepath: Union[str, Path] = "data/processed/road_segments/",
                 length_seg: float = 30,
                 grid_size: float = 0.1,
                 highway_types: List[str] = None,
                 verbose: bool = True):
        
        self.filepath = Path(filepath)
        self.length_seg = length_seg
        self.grid_size = grid_size
        self.highway_types = highway_types
        self.verbose = verbose
        
        self.filepath.mkdir(parents=True, exist_ok=True)
        self.summary = self._load_summary()
        self._build_lookup()
    
    def _round_coord(self, val: float) -> float:
        """Round coordinate to 1 decimal place."""
        return round(val, 1)
    
    def _load_summary(self) -> Optional['gpd.GeoDataFrame']:
        """Load the summary file if it exists."""
        if not HAS_GEOPANDAS:
            return None
        
        summary_path = self.filepath / self.SUMMARY_FILENAME
        
        if not summary_path.exists():
            # Check for legacy files
            legacy_pattern = str(self.filepath / "processed_segments_*.geojson")
            legacy_files = sorted(glob.glob(legacy_pattern))
            
            if legacy_files:
                if self.verbose:
                    print(f"  Migrating from legacy summary files...")
                
                all_summaries = []
                for f in legacy_files:
                    try:
                        df = gpd.read_file(f)
                        all_summaries.append(df)
                    except Exception as e:
                        pass
                
                if all_summaries:
                    merged = pd.concat(all_summaries, ignore_index=True)
                    merged = self._deduplicate_summary(merged)
                    merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{CRS_WGS}")
                    merged.to_file(summary_path, driver='GeoJSON')
                    return merged
            
            return None
        
        if self.verbose:
            print(f"  Loading summary from {summary_path.name}")
        
        try:
            summary = gpd.read_file(summary_path)
            
            if 'length_seg_m' in summary.columns:
                summary = summary[summary['length_seg_m'] == self.length_seg].copy()
            
            if self.verbose:
                print(f"  Found {len(summary)} cached grid cells for {self.length_seg}m segments")
            
            return summary
            
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not read summary file: {e}")
            return None
    
    def _deduplicate_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate grid cells."""
        if len(summary) == 0:
            return summary
    
        summary = summary.copy()
        summary['xmin'] = summary['xmin'].apply(self._round_coord)
        summary['ymin'] = summary['ymin'].apply(self._round_coord)
    
        if 'date_processed' in summary.columns:
            # Ensure consistent string type before sorting
            summary['date_processed'] = summary['date_processed'].astype(str)
            summary = summary.sort_values('date_processed', ascending=False)
    
        summary = summary.drop_duplicates(
            subset=['xmin', 'ymin', 'length_seg_m'], 
            keep='first'
        )
    
        return summary.reset_index(drop=True)
    
    def _build_lookup(self):
        """Build lookup dict for fast cell checking."""
        self._existing_cells = {}
        
        if self.summary is not None and len(self.summary) > 0:
            for _, row in self.summary.iterrows():
                key = (self._round_coord(row['xmin']), 
                       self._round_coord(row['ymin']))
                self._existing_cells[key] = row['filename']
    
    def _get_grid_coords(self, lat: np.ndarray, lon: np.ndarray) -> set:
        """Get unique grid cell coordinates for given points."""
        grid_lons = np.floor(lon / self.grid_size) * self.grid_size
        grid_lats = np.floor(lat / self.grid_size) * self.grid_size
        
        cells = set()
        for x, y in zip(grid_lons, grid_lats):
            cells.add((self._round_coord(x), self._round_coord(y)))
        
        return cells
    
    def _get_segment_filename(self, ymin: float, xmin: float) -> str:
        """Generate filename for a grid cell's segments."""
        lat_str = f"{abs(ymin):.1f}{'N' if ymin >= 0 else 'S'}"
        lon_str = f"{abs(xmin):.1f}{'W' if xmin < 0 else 'E'}"
        return f"segments_{lat_str}_{lon_str}_{int(self.length_seg)}m.geojson"
    
    def _create_grid_cell_polygon(self, xmin: float, ymin: float) -> 'gpd.GeoDataFrame':
        """Create a GeoDataFrame with a single grid cell polygon."""
        poly = create_grid_polygon(xmin, ymin, self.grid_size)
        return gpd.GeoDataFrame({'geometry': [poly]}, crs=f"EPSG:{CRS_WGS}")
    
    def _save_summary(self):
        """Save the current summary to disk."""
        if self.summary is None or len(self.summary) == 0:
            return
        
        summary_path = self.filepath / self.SUMMARY_FILENAME
        
        if not isinstance(self.summary, gpd.GeoDataFrame):
            self.summary = gpd.GeoDataFrame(self.summary, crs=f"EPSG:{CRS_WGS}")
        
        self.summary.to_file(summary_path, driver='GeoJSON')
    
    def _load_single_segment_file(self, filepath: Path) -> Optional['gpd.GeoDataFrame']:
        """Load a single segment file. Returns None on error."""
        try:
            return gpd.read_file(filepath)
        except Exception:
            return None
    
    def _load_segments_parallel(self, files_to_load: List[Path], max_workers: int = 8) -> List['gpd.GeoDataFrame']:
        """Load multiple segment files in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_segment_file, f): f 
                for f in files_to_load
            }
            
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None and len(result) > 0:
                    results.append(result)
        
        return results
    
    def get_road_segments(self,
                          coords_gdf: 'gpd.GeoDataFrame',
                          buffer: float = 100,
                          fill_gaps: bool = True) -> 'gpd.GeoDataFrame':
        """
        Get road segments within buffer distance of input coordinates.
        
        OPTIMIZED VERSION:
        - Parallel file loading
        - Simple bounding box filter instead of expensive spatial operations
        """
        if not HAS_GEOPANDAS:
            raise RuntimeError("geopandas required")
        
        t0 = time.time()
        
        # Get coordinate arrays
        lons = coords_gdf.geometry.x.values
        lats = coords_gdf.geometry.y.values
        
        # 1. Identify grid cells needed
        needed_cells = self._get_grid_coords(lats, lons)
        
        if self.verbose:
            print(f"  GPS data spans {len(needed_cells)} grid cells")
        
        # 2. Find existing vs missing
        existing = {cell for cell in needed_cells if cell in self._existing_cells}
        missing = needed_cells - existing
        
        if self.verbose:
            print(f"  Found {len(existing)} cached, {len(missing)} new")
        
        # 3. Process missing grid cells
        if len(missing) > 0 and fill_gaps:
            if self.verbose:
                print(f"  Processing {len(missing)} new grid cells...")
            self._process_grid_cells(list(missing))
        
        # 4. Build list of files to load
        files_to_load = []
        for cell in needed_cells:
            if cell in self._existing_cells:
                seg_file = self.filepath / self._existing_cells[cell]
                if seg_file.exists():
                    files_to_load.append(seg_file)
        
        if not files_to_load:
            return self._empty_segments_gdf()
        
        # 5. Load segments in parallel
        if self.verbose:
            print(f"  Loading {len(files_to_load)} segment files...", end=" ", flush=True)
        
        t1 = time.time()
        all_segments = self._load_segments_parallel(files_to_load)
        
        if self.verbose:
            print(f"done ({time.time()-t1:.1f}s)")
        
        if not all_segments:
            return self._empty_segments_gdf()
        
        # 6. Concatenate all segments
        if self.verbose:
            print(f"  Concatenating segments...", end=" ", flush=True)
        
        t2 = time.time()
        road_segments = pd.concat(all_segments, ignore_index=True)
        road_segments = gpd.GeoDataFrame(road_segments, crs=f"EPSG:{CRS_WGS}")
        
        if self.verbose:
            print(f"done ({time.time()-t2:.1f}s)")
        
        # 7. Simple bounding box filter (MUCH faster than buffer intersection)
        # Add a small buffer in degrees (roughly buffer meters / 111000)
        buffer_deg = buffer / 111000.0 * 1.5  # 1.5x for safety margin
        
        lon_min, lon_max = lons.min() - buffer_deg, lons.max() + buffer_deg
        lat_min, lat_max = lats.min() - buffer_deg, lats.max() + buffer_deg
        
        # Get segment centers for filtering
        lon_col = f'lon_{int(self.length_seg)}'
        lat_col = f'lat_{int(self.length_seg)}'
        
        if lon_col in road_segments.columns and lat_col in road_segments.columns:
            mask = (
                (road_segments[lon_col] >= lon_min) & 
                (road_segments[lon_col] <= lon_max) &
                (road_segments[lat_col] >= lat_min) & 
                (road_segments[lat_col] <= lat_max)
            )
            road_segments = road_segments[mask]
        
        if self.verbose:
            print(f"  Loaded {len(road_segments)} road segments ({time.time()-t0:.1f}s total)")
        
        return road_segments
    
    def _empty_segments_gdf(self) -> 'gpd.GeoDataFrame':
        """Return empty GeoDataFrame with correct schema."""
        return gpd.GeoDataFrame({
            f'lon_{int(self.length_seg)}': [],
            f'lat_{int(self.length_seg)}': [],
            'rd_type': [],
            'rd_name': [],
            'oneway': [],
            'service': [],
            'rd_deg': [],
            'geometry': []
        }, crs=f"EPSG:{CRS_WGS}")
    
    def _process_grid_cells(self, grid_coords: List[Tuple[float, float]]):
        """Process multiple grid cells and update summary."""
        # Import here to avoid circular imports
        from gps_pipeline import create_road_segments
        
        new_entries = []
        
        for i, (xmin, ymin) in enumerate(grid_coords):
            if self.verbose:
                print(f"    [{i+1}/{len(grid_coords)}] Cell ({ymin:.1f}N, {abs(xmin):.1f}W)...", end=" ", flush=True)
            
            bpoly = self._create_grid_cell_polygon(xmin, ymin)
            
            try:
                segments = create_road_segments(
                    bpoly=bpoly,
                    length_seg=self.length_seg,
                    highway_types=self.highway_types,
                    clip_osm=True,
                    verbose=False
                )
                
                filename = self._get_segment_filename(ymin, xmin)
                seg_path = self.filepath / filename
                segments.to_file(seg_path, driver='GeoJSON')
                
                if self.verbose:
                    print(f"{len(segments)} segments")
                
                new_entries.append({
                    'xmin': self._round_coord(xmin),
                    'ymin': self._round_coord(ymin),
                    'roads_source': 'OSM',
                    'length_seg_m': self.length_seg,
                    'date_processed': date.today().isoformat(),
                    'filename': filename,
                    'geometry': create_grid_polygon(xmin, ymin, self.grid_size)
                })
                
                self._existing_cells[(self._round_coord(xmin), self._round_coord(ymin))] = filename
                
            except Exception as e:
                if self.verbose:
                    print(f"Error: {e}")
                continue
            
            time.sleep(0.5)
        
        if new_entries:
            new_summary = gpd.GeoDataFrame(new_entries, crs=f"EPSG:{CRS_WGS}")
            
            if self.summary is not None and len(self.summary) > 0:
                self.summary = pd.concat([self.summary, new_summary], ignore_index=True)
            else:
                self.summary = new_summary
            
            self.summary = self._deduplicate_summary(self.summary)
            self.summary = gpd.GeoDataFrame(self.summary, crs=f"EPSG:{CRS_WGS}")
            self._save_summary()


def get_road_segments(coords_gdf: 'gpd.GeoDataFrame',
                      filepath: Union[str, Path] = "data/processed/road_segments/",
                      buffer: float = 100,
                      fill_gaps: bool = True,
                      length_seg: float = 30,
                      highway_types: List[str] = None,
                      verbose: bool = True) -> 'gpd.GeoDataFrame':
    """Get road segments within buffer distance of coordinates."""
    manager = RoadSegmentManager(
        filepath=filepath,
        length_seg=length_seg,
        highway_types=highway_types,
        verbose=verbose
    )
    return manager.get_road_segments(coords_gdf, buffer=buffer, fill_gaps=fill_gaps)



def segments_to_dataframe(segments_gdf: 'gpd.GeoDataFrame', 
                          length_seg: float = 30) -> pd.DataFrame:
    """
    Convert GeoDataFrame segments to DataFrame format for matching.
    
    Parameters
    ----------
    segments_gdf : GeoDataFrame
        Road segments from get_road_segments
    length_seg : float
        Segment length (for column naming)
    
    Returns
    -------
    pd.DataFrame
        Segments in format expected by FastRoadMatcher
    """
    lon_col = f'lon_{int(length_seg)}'
    lat_col = f'lat_{int(length_seg)}'
    
    # Extract start/end coordinates from LineString geometry
    start_lats = []
    start_lons = []
    end_lats = []
    end_lons = []
    
    for geom in segments_gdf.geometry:
        if geom is not None and geom.geom_type == 'LineString':
            coords = list(geom.coords)
            start_lons.append(coords[0][0])
            start_lats.append(coords[0][1])
            end_lons.append(coords[-1][0])
            end_lats.append(coords[-1][1])
        else:
            start_lons.append(np.nan)
            start_lats.append(np.nan)
            end_lons.append(np.nan)
            end_lats.append(np.nan)
    
    df = pd.DataFrame({
        'segment_id': range(len(segments_gdf)),
        'road_name': segments_gdf['rd_name'].values if 'rd_name' in segments_gdf else None,
        'highway': segments_gdf['rd_type'].values if 'rd_type' in segments_gdf else None,
        'oneway': segments_gdf['oneway'].values if 'oneway' in segments_gdf else False,
        'start_lat': start_lats,
        'start_lon': start_lons,
        'end_lat': end_lats,
        'end_lon': end_lons,
        'center_lat': segments_gdf[lat_col].values if lat_col in segments_gdf else None,
        'center_lon': segments_gdf[lon_col].values if lon_col in segments_gdf else None,
        'bearing_deg': segments_gdf['rd_deg'].values if 'rd_deg' in segments_gdf else None,
    })
    
    return df


# Legacy function for backward compatibility
def download_and_segment_roads(gps_df: pd.DataFrame, 
                               config: PipelineConfig = None) -> pd.DataFrame:
    """
    Part 2: Download roads from OSM and segment them.
    
    This is the simple (non-caching) version for backward compatibility.
    For the grid-based caching system, use get_road_segments() instead.
    
    Parameters
    ----------
    gps_df : pd.DataFrame
        GPS data (to determine bounding box)
    config : PipelineConfig
    
    Returns
    -------
    pd.DataFrame
        Road segments
    """
    if config is None:
        config = PipelineConfig()
    
    # Get valid coordinates
    valid_lats = gps_df['lat_kf'].dropna().values
    valid_lons = gps_df['lon_kf'].dropna().values
    
    if len(valid_lats) == 0:
        return pd.DataFrame()
    
    # Create bounding box with buffer
    buffer_deg = config.osm_buffer_m / 111000
    bbox = (
        valid_lats.min() - buffer_deg,
        valid_lons.min() - buffer_deg,
        valid_lats.max() + buffer_deg,
        valid_lons.max() + buffer_deg
    )
    
    # Download and segment
    segments_gdf = create_road_segments(
        bbox=bbox,
        length_seg=config.segment_length_m,
        highway_types=config.osm_highway_types,
        clip_osm=False,
        verbose=True
    )
    
    # Convert to DataFrame format
    return segments_to_dataframe(segments_gdf, config.segment_length_m)


# =============================================================================
# Part 3: Fast Road Matching
# =============================================================================

def angular_displacement_vectorized(angle1, angle2, oneway):
    """Vectorized angular displacement between bearings."""
    diff1 = (angle1 - angle2) % 360
    diff2 = (angle2 - angle1) % 360
    displacement = np.minimum(diff1, diff2)
    
    angle2_opp = (angle2 + 180) % 360
    diff1_opp = (angle1 - angle2_opp) % 360
    diff2_opp = (angle2_opp - angle1) % 360
    displacement_opp = np.minimum(diff1_opp, diff2_opp)
    
    displacement = np.where(~oneway, np.minimum(displacement, displacement_opp), displacement)
    return displacement


class FastRoadMatcher:
    """Fast GPS-to-road matching using KD-tree and vectorized operations."""
    
    def __init__(self, segments_df: pd.DataFrame):
        self.segments = segments_df.copy()
        self.n_segments = len(segments_df)
        
        # Extract arrays
        self.seg_start_lats = segments_df['start_lat'].values
        self.seg_start_lons = segments_df['start_lon'].values
        self.seg_end_lats = segments_df['end_lat'].values
        self.seg_end_lons = segments_df['end_lon'].values
        self.seg_center_lats = segments_df['center_lat'].values
        self.seg_center_lons = segments_df['center_lon'].values
        self.seg_bearings = segments_df['bearing_deg'].values
        self.seg_oneway = segments_df['oneway'].values.astype(bool)
        self.seg_ids = segments_df['segment_id'].values
        self.seg_names = segments_df['road_name'].values
        self.seg_highways = segments_df['highway'].values
        
        # Build KD-tree
        ref_lat = np.nanmean(self.seg_center_lats)
        self.m_per_deg_lat = 111000.0
        self.m_per_deg_lon = 111000.0 * np.cos(np.radians(ref_lat))
        
        centers_x = self.seg_center_lons * self.m_per_deg_lon
        centers_y = self.seg_center_lats * self.m_per_deg_lat
        self.kdtree = cKDTree(np.column_stack([centers_x, centers_y]))
    
    def match_points(self, gps_df: pd.DataFrame, config: PipelineConfig = None) -> pd.DataFrame:
        """Match GPS points to road segments."""
        if config is None:
            config = PipelineConfig()
        
        n_points = len(gps_df)
        
        # Output arrays
        matched_segment_ids = np.full(n_points, -1, dtype=np.int32)
        matched_distances = np.full(n_points, np.nan)
        matched_angles = np.full(n_points, np.nan)
        matched_snap_lats = np.full(n_points, np.nan)
        matched_snap_lons = np.full(n_points, np.nan)
        matched_center_lats = np.full(n_points, np.nan)
        matched_center_lons = np.full(n_points, np.nan)
        matched_start_lats = np.full(n_points, np.nan)
        matched_start_lons = np.full(n_points, np.nan)
        matched_end_lats = np.full(n_points, np.nan)
        matched_end_lons = np.full(n_points, np.nan)
        match_types = np.empty(n_points, dtype=object)
        match_types[:] = 'no_match'
        
        # GPS arrays
        point_lats = gps_df['lat_kf'].values
        point_lons = gps_df['lon_kf'].values
        point_bearings = gps_df['car_bearing_deg'].values if 'car_bearing_deg' in gps_df.columns else np.full(n_points, np.nan)
        point_speeds = gps_df['car_speed_mph'].values if 'car_speed_mph' in gps_df.columns else np.zeros(n_points)
        
        # Find valid (non-NaN) points
        valid_mask = ~(np.isnan(point_lats) | np.isnan(point_lons))
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No valid points to match
            result = gps_df.copy()
            result['segment_id'] = np.nan
            result['distance_m'] = np.nan
            result['angle_diff'] = np.nan
            result['snap_lat'] = np.nan
            result['snap_lon'] = np.nan
            result['center_lat'] = np.nan
            result['center_lon'] = np.nan
            result['seg_start_lat'] = np.nan
            result['seg_start_lon'] = np.nan
            result['seg_end_lat'] = np.nan
            result['seg_end_lon'] = np.nan
            result['match_type'] = 'no_match'
            result['road_name'] = np.nan
            result['highway'] = np.nan
            return result
        
        # Convert valid points to KD-tree coordinates
        valid_lats = point_lats[valid_indices]
        valid_lons = point_lons[valid_indices]
        points_x = valid_lons * self.m_per_deg_lon
        points_y = valid_lats * self.m_per_deg_lat
        
        # Query KD-tree only for valid points
        candidate_indices = self.kdtree.query_ball_point(
            np.column_stack([points_x, points_y]),
            r=config.kdtree_radius_m
        )
        
        # Process each valid point
        for i, candidates in enumerate(candidate_indices):
            idx = valid_indices[i]  # Map back to original index
            
            if len(candidates) == 0:
                continue
            
            candidates = np.array(candidates)
            
            # Project point to segments
            distances, snap_lats, snap_lons = self._project_point_to_segments(
                point_lats[idx], point_lons[idx], candidates
            )
            
            within_dist = distances <= config.max_match_distance_m
            if not np.any(within_dist):
                continue
            
            valid_candidates = candidates[within_dist]
            valid_distances = distances[within_dist]
            valid_snap_lats = snap_lats[within_dist]
            valid_snap_lons = snap_lons[within_dist]
            
            valid_angles = angular_displacement_vectorized(
                point_bearings[idx],
                self.seg_bearings[valid_candidates],
                self.seg_oneway[valid_candidates]
            )
            
            use_direction = (
                point_speeds[idx] >= config.min_speed_for_direction_mph and
                not np.isnan(point_bearings[idx])
            )
            
            best_idx = None
            match_type = 'distance_only'
            
            if use_direction:
                within_angle = valid_angles <= config.max_match_angle_deg
                if np.any(within_angle):
                    dir_distances = np.where(within_angle, valid_distances, np.inf)
                    best_idx = np.argmin(dir_distances)
                    match_type = 'direction'
            
            if best_idx is None:
                best_idx = np.argmin(valid_distances)
                match_type = 'distance_only'
            
            best_seg_idx = valid_candidates[best_idx]
            matched_segment_ids[idx] = self.seg_ids[best_seg_idx]
            matched_distances[idx] = valid_distances[best_idx]
            matched_angles[idx] = valid_angles[best_idx]
            matched_snap_lats[idx] = valid_snap_lats[best_idx]
            matched_snap_lons[idx] = valid_snap_lons[best_idx]
            matched_center_lats[idx] = self.seg_center_lats[best_seg_idx]
            matched_center_lons[idx] = self.seg_center_lons[best_seg_idx]
            matched_start_lats[idx] = self.seg_start_lats[best_seg_idx]
            matched_start_lons[idx] = self.seg_start_lons[best_seg_idx]
            matched_end_lats[idx] = self.seg_end_lats[best_seg_idx]
            matched_end_lons[idx] = self.seg_end_lons[best_seg_idx]
            match_types[idx] = match_type
        
        # Build result
        result = gps_df.copy()
        result['segment_id'] = matched_segment_ids
        result['segment_id'] = result['segment_id'].replace(-1, np.nan)
        result['distance_m'] = matched_distances
        result['angle_diff'] = matched_angles
        result['snap_lat'] = matched_snap_lats
        result['snap_lon'] = matched_snap_lons
        result['center_lat'] = matched_center_lats
        result['center_lon'] = matched_center_lons
        result['seg_start_lat'] = matched_start_lats
        result['seg_start_lon'] = matched_start_lons
        result['seg_end_lat'] = matched_end_lats
        result['seg_end_lon'] = matched_end_lons
        result['match_type'] = match_types
        
        # Add road info
        seg_name_map = dict(zip(self.seg_ids, self.seg_names))
        seg_highway_map = dict(zip(self.seg_ids, self.seg_highways))
        result['road_name'] = result['segment_id'].map(seg_name_map)
        result['highway'] = result['segment_id'].map(seg_highway_map)
        
        return result
    
    def _project_point_to_segments(self, point_lat, point_lon, segment_indices):
        """Project point onto multiple segments."""
        start_lats = self.seg_start_lats[segment_indices]
        start_lons = self.seg_start_lons[segment_indices]
        end_lats = self.seg_end_lats[segment_indices]
        end_lons = self.seg_end_lons[segment_indices]
        
        px = (point_lon - start_lons) * self.m_per_deg_lon
        py = (point_lat - start_lats) * self.m_per_deg_lat
        
        seg_dx = (end_lons - start_lons) * self.m_per_deg_lon
        seg_dy = (end_lats - start_lats) * self.m_per_deg_lat
        
        seg_len_sq = seg_dx**2 + seg_dy**2
        seg_len_sq = np.where(seg_len_sq == 0, 1, seg_len_sq)
        
        t = (px * seg_dx + py * seg_dy) / seg_len_sq
        t = np.clip(t, 0, 1)
        
        nearest_x = t * seg_dx
        nearest_y = t * seg_dy
        
        snap_lons = start_lons + nearest_x / self.m_per_deg_lon
        snap_lats = start_lats + nearest_y / self.m_per_deg_lat
        
        distances = np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
        
        return distances, snap_lats, snap_lons


def match_gps_to_roads(gps_df: pd.DataFrame, 
                       segments_df: pd.DataFrame,
                       config: PipelineConfig = None) -> pd.DataFrame:
    """Part 3: Match GPS points to road segments."""
    if config is None:
        config = PipelineConfig()
    
    matcher = FastRoadMatcher(segments_df)
    return matcher.match_points(gps_df, config)


# =============================================================================
# Part 4: Drive Pass Calculation and Export
# =============================================================================

def calculate_drive_passes(df: pd.DataFrame) -> pd.Series:
    """Calculate drive pass IDs (global counter)."""
    n = len(df)
    drive_pass = np.full(n, np.nan)
    center_lats = df['center_lat'].values
    center_lons = df['center_lon'].values
    
    current_pass = 0
    prev_key = None
    
    for i in range(n):
        lat, lon = center_lats[i], center_lons[i]
        if pd.isna(lat) or pd.isna(lon):
            prev_key = None
            continue
        
        seg_key = (round(lat, 6), round(lon, 6))
        if prev_key is None or seg_key != prev_key:
            current_pass += 1
        
        drive_pass[i] = current_pass
        prev_key = seg_key
    
    return pd.Series(drive_pass, index=df.index)


def create_segment_geometries(df: pd.DataFrame) -> 'gpd.GeoDataFrame':
    """
    Create GeoDataFrame with LineString geometries for matched segments.
    
    Each row gets a LineString from (seg_start_lat, seg_start_lon) to (seg_end_lat, seg_end_lon).
    """
    if not HAS_GEOPANDAS:
        raise RuntimeError("geopandas required for geometry creation")
    
    geometries = []
    for _, row in df.iterrows():
        if pd.notna(row.get('seg_start_lat')) and pd.notna(row.get('seg_end_lat')):
            line = LineString([
                (row['seg_start_lon'], row['seg_start_lat']),
                (row['seg_end_lon'], row['seg_end_lat'])
            ])
            geometries.append(line)
        else:
            geometries.append(None)
    
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf


def export_geoparquet(gdf: 'gpd.GeoDataFrame', filepath: str):
    """Export GeoDataFrame to GeoParquet."""
    if not HAS_GEOPANDAS:
        raise RuntimeError("geopandas required for GeoParquet export")
    
    gdf.to_parquet(filepath, index=False)


def format_output(df: pd.DataFrame, include_geometry: bool = True) -> pd.DataFrame:
    """
    Format output DataFrame with standard columns.
    
    Columns:
    - datetime, lat, lon (raw)
    - car_mph, car_deg (speed/bearing)
    - lat_30, lon_30 (segment center, rounded to 6 decimals)
    - rd_type, rd_name (road info)
    - drive_pass
    - lat_smooth, lon_smooth (Kalman)
    - lat_snap, lon_snap (road snapped)
    - seg_start_lat, seg_start_lon, seg_end_lat, seg_end_lon (segment endpoints)
    """
    output = pd.DataFrame()
    
    output['datetime'] = df['datetime']
    output['lat'] = df.get('lat')
    output['lon'] = df.get('lon')
    output['car_mph'] = df.get('car_speed_mph')
    output['car_deg'] = df.get('car_bearing_deg')
    
    # Round segment centers to 6 decimals for consistent keying
    output['lat_30'] = df.get('center_lat')
    output['lon_30'] = df.get('center_lon')
    if output['lat_30'] is not None:
        output['lat_30'] = output['lat_30'].round(6)
    if output['lon_30'] is not None:
        output['lon_30'] = output['lon_30'].round(6)
    
    output['rd_type'] = df.get('highway')
    output['rd_name'] = df.get('road_name')
    output['drive_pass'] = df.get('drive_pass')
    output['lat_smooth'] = df.get('lat_kf')
    output['lon_smooth'] = df.get('lon_kf')
    output['lat_snap'] = df.get('snap_lat')
    output['lon_snap'] = df.get('snap_lon')
    
    if include_geometry:
        output['seg_start_lat'] = df.get('seg_start_lat')
        output['seg_start_lon'] = df.get('seg_start_lon')
        output['seg_end_lat'] = df.get('seg_end_lat')
        output['seg_end_lon'] = df.get('seg_end_lon')
    
    return output


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(gps_filepath: str,
                 segments_filepath: str = None,
                 output_filepath: str = None,
                 config: PipelineConfig = None,
                 use_cached_segments: bool = True,
                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete GPS processing pipeline.
    
    Parameters
    ----------
    gps_filepath : str
        Path to input GPS data (VanDAQ format)
    segments_filepath : str, optional
        Path to pre-computed road segments CSV. If None, uses grid-based caching.
    output_filepath : str, optional
        Path for output file. Extension determines format:
        - .parquet or .geoparquet: GeoParquet with LineString geometries
        - .csv: Standard CSV
    config : PipelineConfig, optional
        Pipeline configuration
    use_cached_segments : bool
        If True, use grid-based segment caching (recommended).
        If False, download fresh from OSM each time.
    verbose : bool
        Print progress
    
    Returns
    -------
    result_df : pd.DataFrame
        Processed GPS data
    segments_df : pd.DataFrame
        Road segments used for matching
    """
    if config is None:
        config = PipelineConfig()
    
    start_time = time.time()
    
    # Part 1: GPS Loading and Smoothing
    if verbose:
        print("Part 1: Loading and smoothing GPS data...")
    gps_df = process_gps_part1(gps_filepath, config)
    if verbose:
        print(f"  Loaded {len(gps_df)} GPS points")
    
    # Part 2: Road Segments
    if segments_filepath and Path(segments_filepath).exists():
        if verbose:
            print(f"Part 2: Loading road segments from {segments_filepath}...")
        
        # Check file extension
        if segments_filepath.endswith('.geojson'):
            segments_gdf = gpd.read_file(segments_filepath)
            segments_df = segments_to_dataframe(segments_gdf, config.segment_length_m)
        else:
            segments_df = pd.read_csv(segments_filepath)
    elif use_cached_segments and HAS_GEOPANDAS:
        if verbose:
            print("Part 2: Loading road segments (with grid-based caching)...")
        
        # Create GeoDataFrame of GPS points for segment lookup
        valid_mask = gps_df['lat_kf'].notna() & gps_df['lon_kf'].notna()
        valid_gps = gps_df[valid_mask].copy()
        
        coords_gdf = gpd.GeoDataFrame(
            valid_gps,
            geometry=gpd.points_from_xy(valid_gps['lon_kf'], valid_gps['lat_kf']),
            crs="EPSG:4326"
        )
        
        # Get segments using caching system
        segments_gdf = get_road_segments(
            coords_gdf,
            filepath=config.roads_filepath,
            buffer=config.kdtree_radius_m,
            fill_gaps=True,
            length_seg=config.segment_length_m,
            highway_types=config.osm_highway_types,
            verbose=verbose
        )
        
        # Filter to named service roads without service descriptors
        if 'service' in segments_gdf.columns:
            service_mask = (segments_gdf['rd_type'] != 'service') | \
                          (segments_gdf['rd_name'].notna() & segments_gdf['service'].isna())
            segments_gdf = segments_gdf[service_mask]
        
        segments_df = segments_to_dataframe(segments_gdf, config.segment_length_m)
    else:
        if verbose:
            print("Part 2: Downloading and segmenting roads from OSM...")
        segments_df = download_and_segment_roads(gps_df, config)
    
    if verbose:
        print(f"  {len(segments_df)} road segments")
    
    # Part 3: Road Matching
    if verbose:
        print("Part 3: Matching GPS points to roads...")
    t0 = time.time()
    matched_df = match_gps_to_roads(gps_df, segments_df, config)
    n_matched = matched_df['segment_id'].notna().sum()
    if verbose:
        print(f"  Matched {n_matched}/{len(matched_df)} ({100*n_matched/len(matched_df):.1f}%) in {time.time()-t0:.2f}s")
    
    # Part 4: Drive Pass Calculation
    if verbose:
        print("Part 4: Calculating drive passes...")
    matched_df['drive_pass'] = calculate_drive_passes(matched_df)
    n_passes = matched_df['drive_pass'].max()
    if verbose:
        print(f"  {n_passes:.0f} drive passes")
    
    # Format output
    result_df = format_output(matched_df, include_geometry=True)
    
    # Export
    if output_filepath:
        if verbose:
            print(f"Exporting to {output_filepath}...")
        
        ext = Path(output_filepath).suffix.lower()
        
        if ext == '.parquet':
            if HAS_GEOPANDAS:
                gdf = create_segment_geometries(result_df)
                # Use .parquet extension (R compatible)
                gdf.to_parquet(output_filepath, index=False)
            else:
                # Fallback: save as parquet without geometry using pyarrow
                try:
                    result_df.to_parquet(output_filepath, index=False)
                except Exception:
                    warnings.warn("Parquet export failed, saving as CSV instead")
                    result_df.to_csv(output_filepath.replace('.parquet', '.csv'), index=False)
        else:
            result_df.to_csv(output_filepath, index=False)
    
    if verbose:
        print(f"\nPipeline complete in {time.time()-start_time:.2f}s")
    
    return result_df, segments_df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPS Processing Pipeline for Mobile Air Quality Monitoring")
    parser.add_argument("input", help="Input GPS data file (VanDAQ CSV)")
    parser.add_argument("-o", "--output", help="Output file path (.parquet, .geoparquet, or .csv)")
    parser.add_argument("-s", "--segments", help="Pre-computed road segments file")
    parser.add_argument("-r", "--roads-dir", default="data/processed/road_segments/",
                        help="Directory for cached road segments")
    parser.add_argument("--max-distance", type=float, default=15.0, help="Max matching distance in meters")
    parser.add_argument("--segment-length", type=float, default=30.0, help="Road segment length in meters")
    parser.add_argument("--no-cache", action="store_true", help="Disable grid-based segment caching")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        max_match_distance_m=args.max_distance,
        segment_length_m=args.segment_length,
        roads_filepath=args.roads_dir
    )
    
    result, segments = run_pipeline(
        args.input,
        segments_filepath=args.segments,
        output_filepath=args.output,
        config=config,
        use_cached_segments=not args.no_cache,
        verbose=not args.quiet
    )
    
    print(f"\nOutput shape: {result.shape}")
    print(f"Matched to road: {result['rd_name'].notna().sum()}")
    print(f"Drive passes: {result['drive_pass'].max():.0f}")