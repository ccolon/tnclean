"""Tests for core geometric operations."""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

from tnclean.types import CleanConfig, ProcessingStats
from tnclean.ops import (
    auto_select_crs,
    explode_multilinestrings,
    snap_coordinates,
    validate_and_clamp_config,
    deoverlap_linestrings,
    simplify_topology_preserving,
    preserve_and_merge_attributes,
)


class TestAutoSelectCRS:
    """Tests for automatic CRS selection."""
    
    def test_auto_select_crs_small_dataset_northern(self):
        """Test UTM selection for small northern hemisphere dataset."""
        # Create small dataset around London (UTM zone 30N expected)
        lines = [LineString([(-0.1, 51.5), (0.1, 51.6)])]  # ~14km
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        crs = auto_select_crs(gdf)
        
        # Should select UTM zone 30N for London area
        assert crs == "EPSG:32630"
    
    def test_auto_select_crs_small_dataset_southern(self):
        """Test UTM selection for small southern hemisphere dataset."""
        # Create small dataset in southern hemisphere (Sydney area)
        lines = [LineString([(151.2, -33.8), (151.3, -33.7)])]  # ~14km
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        crs = auto_select_crs(gdf)
        
        # Should select UTM zone 56S for Sydney area  
        assert crs == "EPSG:32756"
    
    def test_auto_select_crs_polar_north(self):
        """Test UPS selection for Arctic dataset."""
        # Create dataset in Arctic
        lines = [LineString([(0, 85), (1, 85)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        crs = auto_select_crs(gdf)
        
        # Should select UPS North
        assert crs == "EPSG:5041"
    
    def test_auto_select_crs_polar_south(self):
        """Test UPS selection for Antarctic dataset."""
        # Create dataset in Antarctica
        lines = [LineString([(0, -85), (1, -85)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        crs = auto_select_crs(gdf)
        
        # Should select UPS South
        assert crs == "EPSG:5042"
    
    def test_auto_select_crs_large_dataset(self):
        """Test LAEA selection for large dataset."""
        # Create large dataset (>800km diagonal)
        lines = [LineString([(-10, 50), (10, 60)])]  # Large extent across Europe
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        crs = auto_select_crs(gdf)
        
        # Should select custom LAEA
        assert crs.startswith('+proj=laea')
        assert '+lat_0=' in crs
        assert '+lon_0=' in crs
        assert '+units=m' in crs


class TestExplodeMultiLineStrings:
    """Tests for MultiLineString explosion."""
    
    def test_explode_simple_multilinestring(self):
        """Test exploding a simple MultiLineString."""
        multiline = MultiLineString([
            LineString([(0, 0), (1, 0)]),
            LineString([(2, 0), (3, 0)]),
        ])
        gdf = gpd.GeoDataFrame({
            'geometry': [multiline],
            'highway': ['primary'],
            'name': ['Test Road']
        }, crs='EPSG:4326')
        
        stats = ProcessingStats()
        result = explode_multilinestrings(gdf, stats)
        
        # Should have 2 LineStrings now
        assert len(result) == 2
        assert all(geom.geom_type == 'LineString' for geom in result.geometry)
        
        # Attributes should be preserved
        assert all(result['highway'] == 'primary')
        assert all(result['name'] == 'Test Road')
        
        # Check internal tracking fields
        assert '_original_idx' in result.columns
        assert '_part_idx' in result.columns
        
        # Check statistics
        assert stats.input_multilinestring_count == 1
        assert stats.exploded_count == 2
    
    def test_explode_mixed_geometries(self):
        """Test exploding mixed LineString and MultiLineString."""
        line = LineString([(0, 0), (1, 0)])
        multiline = MultiLineString([
            LineString([(2, 0), (3, 0)]),
            LineString([(4, 0), (5, 0)]),
        ])
        
        gdf = gpd.GeoDataFrame({
            'geometry': [line, multiline],
            'highway': ['secondary', 'primary'],
        }, crs='EPSG:4326')
        
        stats = ProcessingStats()
        result = explode_multilinestrings(gdf, stats)
        
        # Should have 3 LineStrings: 1 original + 2 from MultiLineString
        assert len(result) == 3
        
        # Check that original LineString preserved its part_idx=0
        linestring_rows = result[result['_original_idx'] == 0]
        assert len(linestring_rows) == 1
        assert linestring_rows.iloc[0]['_part_idx'] == 0
        
        # Check MultiLineString parts
        multiline_rows = result[result['_original_idx'] == 1]
        assert len(multiline_rows) == 2
        assert set(multiline_rows['_part_idx']) == {0, 1}
    
    def test_explode_empty_input(self):
        """Test exploding empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        
        stats = ProcessingStats()
        result = explode_multilinestrings(gdf, stats)
        
        assert len(result) == 0
        assert stats.input_multilinestring_count == 0
        assert stats.exploded_count == 0


class TestSnapCoordinates:
    """Tests for coordinate snapping."""
    
    def test_snap_simple_coordinates(self):
        """Test basic coordinate snapping."""
        line = LineString([(0.1, 0.2), (1.7, 2.3)])
        gdf = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        stats = ProcessingStats()
        result = snap_coordinates(gdf, config, stats)
        
        # Check that coordinates are snapped to grid centers
        coords = list(result.geometry.iloc[0].coords)
        assert coords[0] == (0.5, 0.5)  # (0 + 0.5) * 1.0
        assert coords[1] == (1.5, 2.5)  # (1 + 0.5) * 1.0, (2 + 0.5) * 1.0
        
        # Should track vertex changes
        assert stats.snapped_vertex_changes == 2  # Both vertices were moved
    
    def test_snap_different_precision(self):
        """Test snapping with different precision."""
        line = LineString([(0.3, 0.7), (2.1, 4.9)])
        gdf = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=2.0)
        
        stats = ProcessingStats()
        result = snap_coordinates(gdf, config, stats)
        
        coords = list(result.geometry.iloc[0].coords)
        # With precision=2.0: (0 + 0.5) * 2.0 = 1.0, (2 + 0.5) * 2.0 = 5.0
        assert coords[0] == (1.0, 1.0)
        assert coords[1] == (3.0, 5.0)  # 2.1 -> grid 1, 4.9 -> grid 2
    
    def test_snap_already_snapped(self):
        """Test snapping coordinates that are already on grid."""
        line = LineString([(0.5, 0.5), (1.5, 1.5)])  # Already on grid centers
        gdf = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        stats = ProcessingStats()
        result = snap_coordinates(gdf, config, stats)
        
        coords = list(result.geometry.iloc[0].coords)
        assert coords[0] == (0.5, 0.5)
        assert coords[1] == (1.5, 1.5)
        
        # No vertices should have changed
        assert stats.snapped_vertex_changes == 0
    
    def test_snap_preserves_attributes(self):
        """Test that snapping preserves all attributes."""
        line = LineString([(0.1, 0.2), (1.1, 1.2)])
        gdf = gpd.GeoDataFrame({
            'geometry': [line],
            'highway': ['primary'],
            'name': ['Test Road'],
            'oneway': [True]
        }, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        stats = ProcessingStats()
        result = snap_coordinates(gdf, config, stats)
        
        assert result.iloc[0]['highway'] == 'primary'
        assert result.iloc[0]['name'] == 'Test Road' 
        assert result.iloc[0]['oneway'] is True


class TestValidateAndClampConfig:
    """Tests for configuration validation and clamping."""
    
    def test_validate_good_config(self):
        """Test validation with good configuration."""
        # Create simple dataset
        lines = [LineString([(0, 0), (10, 0), (20, 0)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        
        config = CleanConfig(
            snap_precision=1.0,
            overlap_tolerance=0.5,
            simplify_tolerance=2.0,
            prune_epsilon=0.1
        )
        
        validated_config, issues = validate_and_clamp_config(gdf, config)
        
        # Should pass validation with no issues
        assert len(issues) == 0
        assert validated_config.snap_precision == config.snap_precision
    
    def test_validate_bad_config_strict(self):
        """Test validation failure in strict mode."""
        lines = [LineString([(0, 0), (1, 0)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        
        config = CleanConfig(
            snap_precision=100.0,  # Way too large
            strict_validation=True
        )
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_and_clamp_config(gdf, config)
    
    def test_validate_bad_config_non_strict(self):
        """Test validation with clamping in non-strict mode."""
        lines = [LineString([(0, 0), (1, 0), (2, 0)])]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        
        config = CleanConfig(
            snap_precision=100.0,  # Too large
            overlap_tolerance=50.0,  # Too large
            strict_validation=False
        )
        
        clamped_config, issues = validate_and_clamp_config(gdf, config)
        
        # Should have issues reported
        assert len(issues) > 0
        
        # Values should be clamped
        assert clamped_config.snap_precision < config.snap_precision
        assert clamped_config.overlap_tolerance < config.overlap_tolerance


class TestSimplifyTopologyPreserving:
    """Tests for topology-preserving simplification."""
    
    def test_simplify_simple_line(self):
        """Test simplification of a line with redundant vertices."""
        # Line with redundant middle vertex
        line = LineString([(0, 0), (1, 0), (2, 0)])  # Middle point is redundant
        gdf = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:3857')
        config = CleanConfig(simplify_tolerance=0.1)
        
        stats = ProcessingStats()
        result = simplify_topology_preserving(gdf, config, stats)
        
        # Should remove redundant vertex
        simplified_coords = list(result.geometry.iloc[0].coords)
        assert len(simplified_coords) <= 3  # Should be same or fewer vertices
        
        # Statistics should be updated
        assert stats.vertices_before_simplify >= stats.vertices_after_simplify
    
    def test_simplify_preserves_attributes(self):
        """Test that simplification preserves attributes."""
        line = LineString([(0, 0), (0.01, 0), (1, 0)])  # Tiny deviation
        gdf = gpd.GeoDataFrame({
            'geometry': [line],
            'highway': ['primary'],
            'name': ['Test Road']
        }, crs='EPSG:3857')
        config = CleanConfig(simplify_tolerance=0.1)
        
        stats = ProcessingStats()
        result = simplify_topology_preserving(gdf, config, stats)
        
        # Attributes should be preserved
        assert result.iloc[0]['highway'] == 'primary'
        assert result.iloc[0]['name'] == 'Test Road'
    
    def test_simplify_with_pruning(self):
        """Test simplification with segment pruning."""
        line = LineString([(0, 0), (100, 0)])  # Long line
        tiny_line = LineString([(200, 0), (200.01, 0)])  # Tiny line to prune
        
        gdf = gpd.GeoDataFrame({'geometry': [line, tiny_line]}, crs='EPSG:3857')
        config = CleanConfig(
            simplify_tolerance=1.0,
            prune_epsilon=0.1  # Should remove tiny line
        )
        
        stats = ProcessingStats()
        result = simplify_topology_preserving(gdf, config, stats)
        
        # Tiny line should be removed
        assert len(result) == 1
        assert stats.pruned_segments == 1


class TestPreserveAndMergeAttributes:
    """Tests for attribute preservation and merging."""
    
    def test_preserve_all_fields(self):
        """Test preserving all fields when preserve_fields=None."""
        rows = [
            {'highway': 'primary', 'name': 'Main St', 'lanes': 2},
            {'highway': 'secondary', 'name': 'Main St', 'lanes': 1},
        ]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=None)
        
        assert merged['highway'] == 'primary,secondary'
        assert merged['name'] == 'Main St'  # Same value, no concatenation needed
        assert merged['lanes'] == '2,1'
    
    def test_preserve_specific_fields(self):
        """Test preserving only specified fields."""
        rows = [
            {'highway': 'primary', 'name': 'Main St', 'lanes': 2, 'surface': 'asphalt'},
            {'highway': 'secondary', 'name': 'Side St', 'lanes': 1, 'surface': 'concrete'},
        ]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=['highway', 'name'])
        
        assert merged['highway'] == 'primary,secondary'
        assert merged['name'] == 'Main St,Side St'
        assert 'lanes' not in merged
        assert 'surface' not in merged
    
    def test_merge_with_quotes(self):
        """Test concatenation with string values containing commas."""
        rows = [
            {'name': 'First, Second Street'},
            {'name': 'Third Avenue'},
        ]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=['name'])
        
        # String with comma should be quoted
        assert merged['name'] == '"First, Second Street",Third Avenue'
    
    def test_merge_numeric_fields(self):
        """Test concatenation of numeric fields."""
        rows = [
            {'lanes': 2, 'speed': 50.0},
            {'lanes': 4, 'speed': 60.0},
        ]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=['lanes', 'speed'])
        
        # Numeric fields should be comma-separated without quotes
        assert merged['lanes'] == '2,4'
        assert merged['speed'] == '50.0,60.0'
    
    def test_merge_none_values(self):
        """Test handling of None values in fields."""
        rows = [
            {'highway': 'primary', 'name': None},
            {'highway': 'secondary', 'name': 'Test St'},
        ]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=['highway', 'name'])
        
        assert merged['highway'] == 'primary,secondary'
        assert merged['name'] == 'Test St'  # None values should be ignored
    
    def test_merge_empty_rows(self):
        """Test merging with empty row list."""
        merged = preserve_and_merge_attributes([], preserve_fields=['highway'])
        assert merged == {}
    
    def test_merge_single_row(self):
        """Test merging with single row."""
        rows = [{'highway': 'primary', 'name': 'Main St'}]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=['highway', 'name'])
        
        assert merged['highway'] == 'primary'
        assert merged['name'] == 'Main St'
    
    def test_merge_identical_values(self):
        """Test merging rows with identical values."""
        rows = [
            {'highway': 'primary', 'name': 'Main St'},
            {'highway': 'primary', 'name': 'Main St'},
        ]
        
        merged = preserve_and_merge_attributes(rows, preserve_fields=['highway', 'name'])
        
        # Identical values should not be duplicated
        assert merged['highway'] == 'primary'
        assert merged['name'] == 'Main St'