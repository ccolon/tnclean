# tnclean — Transport Network Cleaner

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small Python package to **clean & simplify transport network lines**.  
**Input:** Shapefile or GeoJSON with `LineString`/`MultiLineString`.  
**Output:** **GeoJSON (RFC 7946)** with **LineString-only**, de-duplicated, split at intersections, and **topology-preservingly simplified** geometries.

## Why

Real-world network layers (roads/rail/sea links) often contain MultiLineStrings, near-duplicates, partial overlaps, and unsnapped intersections. These break downstream routing, loading, and ABM pipelines. **tnclean** applies a fixed, topology-first pipeline to produce a consistent, minimal, **LineString-only** network you can trust.

## Features

- **Explode** MultiLineStrings → LineStrings
- **Snap** coordinates to a metric grid to stabilize topology
- **Split** edges at true intersections (edge–edge, node–edge)
- **De-overlap**: detect partial overlaps and split/deduplicate intelligently
- **Merge degree-2 nodes**: join LineStrings connected at nodes with exactly 2 connections
- **Remove small components**: filter out isolated LineString fragments below size thresholds
- **Simplify (topology-preserving)**: global TopoJSON-style arcs with node anchoring
- **Attributes**: inherit on split, concatenate unique values on merge
- **LineString-only GeoJSON** export + detailed **report JSON**

## Installation

**Requirements:** Python 3.11+

```bash
# Install geospatial dependencies with conda/mamba (recommended)
conda install -c conda-forge geopandas shapely>=2 pyproj rtree

# Install tnclean
pip install tnclean

# Or install in development mode
git clone <repo>
cd tnclean
pip install -e ".[dev]"
```

## Quick Start

### Full Pipeline (Recommended)

```bash
# Complete processing pipeline (includes small component removal by default)
tnclean clean input.geojson output.geojson \
  --metric-crs auto \
  --snap-precision 1.0 \
  --overlap-tolerance 0.5 \
  --simplify-tolerance 2.0 \
  --progress

# Disable small component removal to preserve all input data
tnclean clean input.geojson output.geojson \
  --keep-small-components \
  --progress
```

### Modular Operations (For Debugging)

**tnclean** now supports running individual pipeline steps for debugging and analysis:

```bash
# 1. Explode MultiLineStrings to LineStrings
tnclean explode input.geojson exploded.geojson -v

# 2. Snap coordinates to metric grid  
tnclean snap exploded.geojson snapped.geojson --snap-precision 1.0 -v

# 3. Split edges at intersections
tnclean split snapped.geojson split.geojson --snap-precision 1.0 -v

# 4. De-overlap and deduplicate
tnclean deoverlap split.geojson deoverlapped.geojson \
  --overlap-tolerance 0.5 --dedup-policy keep-shortest -v

# 5. Merge degree-2 nodes (optional)
tnclean merge-degree2 deoverlapped.geojson merged.geojson \
  --merge-policy concatenate --max-length-ratio 10.0 -v

# 6. Remove small components (optional)
tnclean remove-components merged.geojson cleaned.geojson \
  --min-features 2 --min-length 10.0 -v

# 7. Topology-preserving simplification
tnclean simplify cleaned.geojson final.geojson \
  --simplify-tolerance 2.0 --prune-epsilon 0.1 -v
```

Each step generates a detailed JSON report with `--report output_report.json`.

### Python API

```python
from tnclean import clean_network, CleanConfig

# Full pipeline
config = CleanConfig(
    metric_crs="auto",
    snap_precision=1.0,
    overlap_tolerance=0.5,
    simplify_tolerance=2.0,
    preserve_fields=["highway", "name"]  # specify fields to keep
)

gdf, report = clean_network("input.geojson", "output.geojson", config)
print(f"Processed {report['input_count']} → {report['output_count']} features")

# Individual operations
from tnclean.modules import (
    run_explode, run_snap, run_split, run_deoverlap, 
    run_merge_degree2, run_remove_components, run_simplify
)

# Step-by-step processing for debugging
gdf1, report1 = run_explode("input.geojson", "exploded.geojson", config)
gdf2, report2 = run_snap("exploded.geojson", "snapped.geojson", config) 
gdf3, report3 = run_split("snapped.geojson", "split.geojson", config)
gdf4, report4 = run_deoverlap("split.geojson", "deoverlapped.geojson", config)
gdf5, report5 = run_merge_degree2("deoverlapped.geojson", "merged.geojson", config)
gdf6, report6 = run_remove_components("merged.geojson", "cleaned.geojson", config)
gdf7, report7 = run_simplify("cleaned.geojson", "final.geojson", config)
```

## Pipeline

**Fixed pipeline order:** explode → snap → split → de-overlap → merge degree-2 → remove small components → simplify → export

1. **Explode**: MultiLineString → LineString, attributes preserved
2. **Snap**: quantize coordinates to metric grid for topology stability
3. **Split**: create nodes at true intersections, split edges between nodes
4. **De-overlap**: detect partial overlaps, split at overlap boundaries, deduplicate
5. **Merge degree-2**: merge LineStrings connected at nodes with exactly 2 connections
6. **Remove small components**: filter out isolated LineString fragments below size thresholds
7. **Simplify**: topology-preserving simplification with global arc sharing
8. **Export**: LineString-only GeoJSON + comprehensive report

### Degree-2 Node Merging

Degree-2 nodes are endpoints that connect exactly 2 LineStrings. These often represent artificial breaks in what should be continuous features (roads, rivers, etc.). The degree-2 merging step:

- **Identifies** nodes with exactly 2 connected LineStrings
- **Validates** geometric connectivity and length ratios
- **Merges** connected segments into single LineStrings
- **Preserves** attributes using configurable policies

This step is **enabled by default** to produce cleaner networks. Disable with `merge_degree2_nodes=False` or use the `tnclean merge-degree2` command for standalone processing.

### Small Component Removal

Small component removal identifies and filters out isolated LineString fragments that are below configurable size thresholds. This helps clean up noisy datasets with small disconnected segments. The algorithm:

- **Groups** features into connected components using Breadth-First Search (BFS)
- **Applies dual thresholds**: components must meet BOTH feature count AND total length criteria
- **Preserves** components that meet: `features >= min_component_features AND length >= min_component_length`
- **Removes** isolated fragments, stray segments, and tiny disconnected pieces

**Configuration options:**
- `remove_small_components`: Enable/disable the feature (default: True)
- `min_component_features`: Minimum features per component to preserve (default: 1)
- `min_component_length`: Minimum total length in meters per component to preserve (default: 5.0)

**Example use cases:**
- Remove stray GPS traces from OpenStreetMap extracts
- Filter out digitization artifacts in manual datasets
- Clean up small disconnected road segments in rural areas
- Remove tiny fragments created by geometric processing

This step is **enabled by default** to produce cleaner networks. Disable with `--keep-small-components` in CLI or `remove_small_components=False` in Python API.

## Configuration

All distance tolerances are in **meters**. The `metric_crs="auto"` option automatically selects an appropriate projected coordinate system.

Key parameters:
- `snap_precision`: Grid size for coordinate quantization (default: "auto")
- `overlap_tolerance`: Distance threshold for overlap detection (default: 5.0m)
- `simplify_tolerance`: Geometric simplification threshold (default: 2.0m)
- `prune_epsilon`: Minimum segment length to preserve (default: 0.1m)
- `preserve_fields`: List of attribute fields to preserve (None = all)
- `dedup_policy`: How to resolve duplicates ("keep-shortest", "keep-first", "keep-longest")
- `check_geometry_paths`: Enable path-based deduplication instead of endpoint-only (default: False)
- `merge_degree2_nodes`: Enable merging of degree-2 nodes (default: True)
- `merge_attribute_policy`: How to merge attributes ("concatenate", "keep-longest", "keep-first")
- `merge_max_length_ratio`: Maximum length ratio between merged segments (default: 10.0)
- `remove_small_components`: Enable removal of small isolated components (default: True)
- `min_component_features`: Minimum features per component to preserve (default: 1)
- `min_component_length`: Minimum total length (meters) per component to preserve (default: 5.0)

## Debugging and Troubleshooting

### Using Modular Commands

The modular commands are particularly useful for:

1. **Isolating problems**: Run individual steps to identify where issues occur
2. **Parameter tuning**: Test different tolerance values on specific operations
3. **Incremental debugging**: Inspect intermediate results at each step
4. **Performance analysis**: Time individual operations separately

**Example debugging workflow:**

```bash
# Start with a small test file
tnclean explode problem_network.geojson step1.geojson -v --report report1.json

# Check if intersection splitting works correctly  
tnclean split step1.geojson step2.geojson --snap-precision 5.0 -vv --report report2.json

# Verify overlap detection with different tolerance
tnclean deoverlap step2.geojson step3.geojson --overlap-tolerance 2.0 -vv --report report3.json

# Test component removal with different thresholds
tnclean remove-components step3.geojson step4.geojson --min-features 1 --min-length 5.0 -v --report report4.json

# Inspect reports to understand what happened
cat report3.json | jq '.overlap_pairs_found, .duplicates_removed'
cat report4.json | jq '.small_components_found, .small_components_removed, .output_connected_components'
```

### Common Issues

- **Too many intersections**: Reduce `snap_precision` or check input data quality
- **Missing intersections**: Increase `snap_precision` or check coordinate precision
- **Over-simplification**: Reduce `simplify_tolerance` or use topology-preserving mode
- **Performance issues**: Process smaller regions or increase `snap_precision` for speed

### Validation

The tool provides data-driven validation warnings with recommended parameter ranges:

```bash
tnclean validate input.geojson  # Get parameter recommendations
```

## Output Format

### GeoJSON Output
- **LineString-only geometries** (no MultiLineStrings)
- **RFC 7946 compliant** (WGS84 coordinates)
- **Preserved attributes** from original features
- **Clean topology** with proper intersections

### Reports
Each operation generates a detailed JSON report with processing statistics:

```json
{
  "operation": "split", 
  "input_count": 150,
  "output_count": 324,
  "split_count": 174,
  "processing_time": 2.34,
  "processing_crs": "EPSG:32633",
  "intersections_found": 87,
  "timestamp": "2024-01-15 14:30:25 UTC"
}
```

**Key metrics by operation:**
- **Explode**: `exploded_count`, `multilinestring_count`
- **Snap**: `snapped_vertex_changes`, `coordinate_precision_used`
- **Split**: `split_count`, `intersections_found`  
- **Deoverlap**: `overlap_pairs_found`, `duplicates_removed`
- **Merge degree-2**: `degree2_nodes_found`, `degree2_merges_performed`
- **Remove components**: `small_components_found`, `small_components_removed`, `small_component_features_removed`
- **Simplify**: `vertices_removed`, `vertex_reduction_percent`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please run the full test suite before submitting:

```bash
ruff format && ruff check && mypy && pytest
```