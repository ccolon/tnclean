"""Tests for the main pipeline orchestration."""

import json
import tempfile
from pathlib import Path

import pytest
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

from tnclean import clean_network, CleanConfig


class TestCleanNetworkPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_simple_pipeline(self):
        """Test complete pipeline with simple data."""
        # Create test data
        lines = [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
        ]
        gdf = gpd.GeoDataFrame({
            'geometry': lines,
            'highway': ['primary', 'secondary'],
        }, crs='EPSG:4326')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            # Save input data
            gdf.to_file(input_path, driver='GeoJSON')
            
            # Run pipeline
            config = CleanConfig(
                snap_precision=0.1,
                overlap_tolerance=0.05,
                simplify_tolerance=0.1,
                progress_bar=False,  # Disable for testing
                verbose=0
            )
            
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Check that output file was created
            assert output_path.exists()
            
            # Check result GeoDataFrame
            assert len(result_gdf) >= 1  # Should have at least some output
            assert all(geom.geom_type == 'LineString' for geom in result_gdf.geometry)
            
            # Check report structure
            assert 'input_count' in report
            assert 'output_count' in report
            assert 'processing_time' in report
            assert 'config' in report
            assert report['input_count'] == 2
    
    def test_pipeline_with_multilinestring(self):
        """Test pipeline with MultiLineString input."""
        multiline = MultiLineString([
            LineString([(0, 0), (1, 0)]),
            LineString([(2, 0), (3, 0)]),
        ])
        
        gdf = gpd.GeoDataFrame({
            'geometry': [multiline],
            'highway': ['trunk'],
        }, crs='EPSG:4326')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(progress_bar=False, verbose=0)
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Should explode MultiLineString
            assert report['input_multilinestring_count'] == 1
            assert report['exploded_count'] == 2
            assert all(geom.geom_type == 'LineString' for geom in result_gdf.geometry)
    
    def test_pipeline_with_duplicates(self):
        """Test pipeline with duplicate geometries."""
        # Create lines that should be duplicates after snapping
        lines = [
            LineString([(0.01, 0.01), (1.01, 0.01)]),
            LineString([(0.02, 0.02), (1.02, 0.02)]),  # Should snap to same as first
            LineString([(2, 0), (3, 0)]),  # Different line
        ]
        
        gdf = gpd.GeoDataFrame({
            'geometry': lines,
            'highway': ['primary', 'primary', 'secondary'],
        }, crs='EPSG:3857')  # Use projected CRS
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(
                snap_precision=0.1,  # Large enough to merge the lines
                dedup_policy='keep-shortest',
                progress_bar=False,
                verbose=0
            )
            
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Should remove duplicates
            assert len(result_gdf) < len(gdf)  # Fewer output lines
            assert report['duplicates_removed'] >= 1
    
    def test_pipeline_auto_crs(self):
        """Test pipeline with automatic CRS selection."""
        # Create data in geographic coordinates (London area)
        lines = [
            LineString([(-0.1, 51.5), (0.1, 51.6)]),
        ]
        
        gdf = gpd.GeoDataFrame({
            'geometry': lines,
            'highway': ['primary'],
        }, crs='EPSG:4326')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(
                metric_crs='auto',
                progress_bar=False,
                verbose=0
            )
            
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Should have processed in metric CRS but output in original
            assert report['original_crs'] == 'EPSG:4326'
            assert 'EPSG:326' in report['processing_crs']  # UTM zone
            assert result_gdf.crs.to_epsg() == 4326  # Output should be WGS84
    
    def test_pipeline_preserve_fields(self):
        """Test pipeline with field preservation."""
        lines = [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
        ]
        
        gdf = gpd.GeoDataFrame({
            'geometry': lines,
            'highway': ['primary', 'secondary'],
            'name': ['Main St', 'Side St'],
            'lanes': [2, 1],
            'surface': ['asphalt', 'concrete'],
        }, crs='EPSG:4326')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(
                preserve_fields=['highway', 'name'],
                progress_bar=False,
                verbose=0
            )
            
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Should preserve specified fields
            assert 'highway' in result_gdf.columns
            assert 'name' in result_gdf.columns
            # Should not preserve unspecified fields
            assert 'lanes' not in result_gdf.columns
            assert 'surface' not in result_gdf.columns
    
    def test_pipeline_invalid_input(self):
        """Test pipeline with invalid input."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'nonexistent.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            config = CleanConfig(progress_bar=False, verbose=0)
            
            with pytest.raises(FileNotFoundError):
                clean_network(str(input_path), str(output_path), config)
    
    def test_pipeline_empty_input(self):
        """Test pipeline with empty input."""
        gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(progress_bar=False, verbose=0)
            
            with pytest.raises(ValueError, match="Input dataset is empty"):
                clean_network(str(input_path), str(output_path), config)
    
    def test_pipeline_no_crs(self):
        """Test pipeline with input data missing CRS."""
        lines = [LineString([(0, 0), (1, 0)])]
        gdf = gpd.GeoDataFrame({'geometry': lines})  # No CRS
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(progress_bar=False, verbose=0)
            
            with pytest.raises(ValueError, match="Input dataset has no CRS defined"):
                clean_network(str(input_path), str(output_path), config)
    
    def test_pipeline_strict_validation(self):
        """Test pipeline with strict validation mode."""
        lines = [LineString([(0, 0), (1, 0)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(
                snap_precision=1000.0,  # Way too large
                strict_validation=True,
                progress_bar=False,
                verbose=0
            )
            
            with pytest.raises(ValueError, match="Configuration validation failed"):
                clean_network(str(input_path), str(output_path), config)
    
    def test_pipeline_report_structure(self):
        """Test that pipeline report has expected structure."""
        lines = [
            LineString([(0, 0), (1, 0), (1.5, 0), (2, 0)]),  # Line with redundant vertex
        ]
        
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(progress_bar=False, verbose=0)
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Check all expected report fields
            expected_fields = [
                'input_count', 'output_count', 'input_total_length', 'output_total_length',
                'length_change', 'length_change_percent', 'exploded_count',
                'snapped_vertex_changes', 'split_count', 'overlap_pairs_found',
                'duplicates_removed', 'vertices_before_simplify', 'vertices_after_simplify',
                'vertices_removed', 'vertex_reduction_percent', 'pruned_segments',
                'processing_time', 'original_crs', 'processing_crs', 'config',
                'pipeline_version', 'timestamp'
            ]
            
            for field in expected_fields:
                assert field in report, f"Missing report field: {field}"
            
            # Check that config is properly serialized
            assert 'snap_precision' in report['config']
            assert 'overlap_tolerance' in report['config']
            
            # Check that report can be JSON serialized
            json.dumps(report, default=str)  # Should not raise
    
    def test_pipeline_output_geojson_rfc7946(self):
        """Test that output is RFC 7946 compliant GeoJSON."""
        lines = [LineString([(0, 0), (1, 0)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')  # Non-WGS84 input
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / 'input.geojson'
            output_path = Path(tmp_dir) / 'output.geojson'
            
            gdf.to_file(input_path, driver='GeoJSON')
            
            config = CleanConfig(progress_bar=False, verbose=0)
            result_gdf, report = clean_network(str(input_path), str(output_path), config)
            
            # Load output file to check CRS
            output_gdf = gpd.read_file(output_path)
            
            # Should be in WGS84 for RFC 7946 compliance
            assert output_gdf.crs.to_epsg() == 4326
            
            # Check JSON structure
            with open(output_path, 'r') as f:
                geojson = json.load(f)
            
            assert geojson['type'] == 'FeatureCollection'
            assert 'features' in geojson
            
            # Features should have LineString geometries
            for feature in geojson['features']:
                assert feature['geometry']['type'] == 'LineString'