"""Property-based tests using Hypothesis."""

import pytest
import geopandas as gpd
from hypothesis import given, strategies as st, assume, settings
from shapely.geometry import LineString

from tnclean.types import CleanConfig
from tnclean.graph import QCKGraph, compute_dataset_stats
from tnclean.ops import explode_multilinestrings, snap_coordinates, preserve_and_merge_attributes
from .conftest import valid_linestring, valid_multilinestring, clean_config_strategy


class TestPropertyBasedGraph:
    """Property-based tests for graph operations."""
    
    @given(valid_linestring())
    @settings(max_examples=50)
    def test_qck_node_identity_deterministic(self, linestring):
        """Test that QCK node identity is deterministic."""
        config = CleanConfig(snap_precision=1.0)
        graph1 = QCKGraph(config, "EPSG:3857")
        graph2 = QCKGraph(config, "EPSG:3857")
        
        # Add same linestring to both graphs
        edge_key1 = graph1.add_edge(linestring, 0)
        edge_key2 = graph2.add_edge(linestring, 0)
        
        # Should produce identical results
        assert edge_key1 == edge_key2
        assert len(graph1._nodes) == len(graph2._nodes)
    
    @given(valid_linestring(), st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=50)
    def test_snap_precision_bounds(self, linestring, precision):
        """Test that snapped coordinates respect precision bounds."""
        config = CleanConfig(snap_precision=precision)
        graph = QCKGraph(config, "EPSG:3857")
        
        snapped = graph.snap_linestring(linestring)
        
        # All coordinates should be at grid centers
        for x, y in snapped.coords:
            # Should be of form (i + 0.5) * precision
            grid_x = (x / precision) - 0.5
            grid_y = (y / precision) - 0.5
            assert abs(grid_x - round(grid_x)) < 1e-10
            assert abs(grid_y - round(grid_y)) < 1e-10
    
    @given(valid_linestring())
    @settings(max_examples=50)
    def test_edge_key_symmetry(self, linestring):
        """Test that edge keys are symmetric (order-independent)."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Add edge
        edge_key1 = graph.add_edge(linestring, 0)
        
        # Add reversed edge
        reversed_coords = list(linestring.coords)
        reversed_coords.reverse()
        reversed_linestring = LineString(reversed_coords)
        edge_key2 = graph.add_edge(reversed_linestring, 1)
        
        # Should be same edge key (after snapping, endpoints should be identical)
        # Note: This might not always be true due to snapping effects, so we check
        # that the edge keys at least have the same nodes (possibly reordered)
        assert {edge_key1.node_a, edge_key1.node_b} == {edge_key2.node_a, edge_key2.node_b}


class TestPropertyBasedOps:
    """Property-based tests for core operations."""
    
    @given(valid_multilinestring())
    @settings(max_examples=50)
    def test_explode_preserves_total_length(self, multilinestring):
        """Test that exploding preserves total length."""
        from tnclean.types import ProcessingStats
        
        gdf = gpd.GeoDataFrame({'geometry': [multilinestring]}, crs='EPSG:4326')
        original_length = multilinestring.length
        
        stats = ProcessingStats()
        result = explode_multilinestrings(gdf, stats)
        
        result_length = sum(geom.length for geom in result.geometry)
        
        # Should preserve total length (within floating point precision)
        assert abs(original_length - result_length) < 1e-10
    
    @given(valid_linestring(), st.floats(min_value=0.01, max_value=100.0))
    @settings(max_examples=50)
    def test_snap_coordinates_idempotent(self, linestring, precision):
        """Test that snapping is idempotent (snapping twice = snapping once)."""
        from tnclean.types import ProcessingStats
        
        gdf = gpd.GeoDataFrame({'geometry': [linestring]}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=precision)
        
        # Snap once
        stats1 = ProcessingStats()
        result1 = snap_coordinates(gdf, config, stats1)
        
        # Snap again
        stats2 = ProcessingStats()  
        result2 = snap_coordinates(result1, config, stats2)
        
        # Second snapping should not change anything
        assert stats2.snapped_vertex_changes == 0
        
        # Geometries should be identical
        geom1 = result1.geometry.iloc[0]
        geom2 = result2.geometry.iloc[0]
        assert geom1.equals_exact(geom2, tolerance=1e-10)
    
    @given(st.lists(st.dictionaries(
        st.text(min_size=1, max_size=10), 
        st.one_of(st.text(min_size=1, max_size=20), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=5
    ), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_preserve_attributes_deterministic(self, rows):
        """Test that attribute preservation is deterministic."""
        assume(len(rows) > 0)
        assume(all(len(row) > 0 for row in rows))
        
        # Get field names that exist in all rows
        common_fields = set(rows[0].keys())
        for row in rows[1:]:
            common_fields &= set(row.keys())
        
        if not common_fields:
            return  # Skip if no common fields
        
        preserve_fields = list(common_fields)
        
        result1 = preserve_and_merge_attributes(rows, preserve_fields)
        result2 = preserve_and_merge_attributes(rows, preserve_fields)
        
        # Should be deterministic
        assert result1 == result2
    
    @given(clean_config_strategy())
    @settings(max_examples=20)
    def test_config_validation_consistency(self, config):
        """Test that valid configs remain valid through operations."""
        # If we can construct the config, it should pass validation
        assert config.snap_precision > 0
        assert config.overlap_tolerance > 0
        assert config.simplify_tolerance >= 0
        assert config.prune_epsilon >= 0


class TestPropertyBasedDatasetStats:
    """Property-based tests for dataset statistics."""
    
    @given(st.lists(valid_linestring(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_dataset_stats_non_negative(self, linestrings):
        """Test that dataset statistics are non-negative."""
        # Filter out degenerate linestrings
        valid_linestrings = [ls for ls in linestrings if ls.length > 0]
        assume(len(valid_linestrings) > 0)
        
        gdf = gpd.GeoDataFrame({'geometry': valid_linestrings}, crs='EPSG:4326')
        
        vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
        
        # All values should be non-negative
        assert all(spacing >= 0 for spacing in vertex_spacings)
        assert all(length >= 0 for length in edge_lengths)
        assert bbox_diagonal >= 0
        
        # Should have reasonable number of measurements
        assert len(edge_lengths) == len(valid_linestrings)
    
    @given(valid_linestring())
    @settings(max_examples=50)
    def test_single_linestring_stats(self, linestring):
        """Test dataset stats for single linestring."""
        assume(linestring.length > 0)
        
        gdf = gpd.GeoDataFrame({'geometry': [linestring]}, crs='EPSG:4326')
        
        vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
        
        # Should have one edge length
        assert len(edge_lengths) == 1
        assert edge_lengths[0] == pytest.approx(linestring.length, rel=1e-10)
        
        # Number of vertex spacings should be vertices - 1
        coords = list(linestring.coords)
        expected_spacings = len(coords) - 1
        assert len(vertex_spacings) == expected_spacings


class TestPropertyBasedInvariants:
    """Property-based tests for important invariants."""
    
    @given(valid_linestring(), st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=50)
    def test_snapping_preserves_topology(self, linestring, precision):
        """Test that snapping preserves basic topological properties."""
        assume(linestring.length > 0)
        
        config = CleanConfig(snap_precision=precision)
        graph = QCKGraph(config, "EPSG:3857")
        
        original_coords = list(linestring.coords)
        snapped = graph.snap_linestring(linestring)
        snapped_coords = list(snapped.coords)
        
        # Should preserve number of vertices
        assert len(original_coords) == len(snapped_coords)
        
        # Should preserve LineString validity
        assert snapped.is_valid
        assert snapped.geom_type == 'LineString'
    
    @given(st.lists(valid_linestring(), min_size=2, max_size=5))
    @settings(max_examples=30)
    def test_graph_node_consistency(self, linestrings):
        """Test that graph node management is consistent."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Add all linestrings
        for i, linestring in enumerate(linestrings):
            if linestring.length > 0:  # Skip degenerate
                graph.add_edge(linestring, i)
        
        # Every edge should connect exactly two nodes
        for edge_key in graph._edges:
            assert edge_key.node_a != edge_key.node_b  # No self-loops
            assert edge_key.node_a in graph._nodes
            assert edge_key.node_b in graph._nodes
        
        # Node degrees should be consistent
        for node_id in graph._nodes:
            degree = graph.get_node_degree(node_id)
            incident_edges = graph.get_incident_edges(node_id)
            assert degree == len(incident_edges)