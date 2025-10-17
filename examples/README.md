# tnclean Examples

This directory contains example data and configuration files for testing tnclean.

## Files

### `simple_network.geojson`
A small test network with common issues that tnclean addresses:
- **MultiLineString** geometry (Highway 101)
- **Near-duplicate** features (Main Street appears twice with slight coordinate differences)
- **Intersection** between Main Street and Cross Street
- **Mixed attribute values** for testing field preservation

### `config.yaml`
Example configuration file showing all available options with comments.

## Usage Examples

### Basic Usage
Clean the example network with default settings:
```bash
tnclean simple_network.geojson output.geojson
```

### Using Configuration File
Use the provided configuration file:
```bash
tnclean simple_network.geojson output.geojson --config config.yaml
```

### Custom Parameters
Override specific parameters:
```bash
tnclean simple_network.geojson output.geojson \
  --snap-precision 0.5 \
  --preserve-field highway \
  --preserve-field name \
  --verbose
```

### Generate Report
Save a detailed processing report:
```bash
tnclean simple_network.geojson output.geojson \
  --config config.yaml \
  --report report.json \
  --verbose
```

### Validation Only
Validate input data without processing:
```bash
tnclean validate simple_network.geojson --verbose
```

## Expected Results

When processing `simple_network.geojson`, you should see:

1. **MultiLineString explosion**: Highway 101 split into 2 LineString features
2. **Coordinate snapping**: All coordinates quantized to the specified grid
3. **Duplicate detection**: The two "Main Street" features identified as duplicates
4. **Deduplication**: One "Main Street" kept based on the dedup policy
5. **Field preservation**: Only specified fields (highway, name, lanes) retained
6. **LineString-only output**: All geometries are simple LineStrings

The final output should contain approximately 3-4 LineString features instead of the original 4 features with mixed geometry types.

## Creating Your Own Examples

To test tnclean with your own data:

1. Ensure your data is in Shapefile or GeoJSON format
2. Ensure it contains LineString and/or MultiLineString geometries
3. Include a coordinate reference system (CRS)
4. Run validation first to check for issues:
   ```bash
   tnclean validate your_data.geojson --verbose
   ```

## Troubleshooting

### Common Issues

**"Input dataset has no CRS defined"**
- Add a CRS to your data or specify one with GIS software

**"Configuration validation failed"**
- Your tolerances may be too large/small for the dataset
- Try `--strict false` to auto-clamp values
- Check the validation output for recommended ranges

**"Empty result"**  
- Your tolerances may be filtering out all features
- Try larger `prune_epsilon` or smaller `simplify_tolerance`

**"Invalid geometry types"**
- tnclean only supports LineString and MultiLineString
- Use GIS software to filter or convert other geometry types