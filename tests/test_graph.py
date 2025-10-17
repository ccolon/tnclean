"""Tests for QCK graph and node identity system."""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point

from tnclean.types import CleanConfig, NodeId, EdgeKey
from tnclean.graph import QCKGraph, compute_dataset_stats, analyze_network_topology


class TestQCKGraph:
    """Tests for QCKGraph class."""
    
    def test_quantize_point(self):
        """Test coordinate quantization."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Test exact grid points
        node1 = graph.quantize_point(0.0, 0.0)
        assert node1.ix == 0
        assert node1.iy == 0
        
        # Test points that should round down
        node2 = graph.quantize_point(0.4, 0.3)
        assert node2.ix == 0
        assert node2.iy == 0
        
        # Test points that should round to next grid cell
        node3 = graph.quantize_point(1.5, 2.7)
        assert node3.ix == 1
        assert node3.iy == 2
        
        # Test negative coordinates
        node4 = graph.quantize_point(-1.2, -0.8)
        assert node4.ix == -2
        assert node4.iy == -1
    
    def test_snap_coordinate(self):
        """Test coordinate snapping to grid centers."""
        config = CleanConfig(snap_precision=2.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Points should snap to grid center (ix + 0.5) * precision
        x, y = graph.snap_coordinate(0.5, 1.5)
        assert x == 1.0  # (0 + 0.5) * 2.0
        assert y == 1.0  # (0 + 0.5) * 2.0
        
        x, y = graph.snap_coordinate(2.1, 3.9)
        assert x == 3.0  # (1 + 0.5) * 2.0
        assert y == 3.0  # (1 + 0.5) * 2.0
    
    def test_add_node(self):
        """Test node addition and retrieval."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Add node
        node_id = graph.add_node(0.3, 0.7)
        assert node_id.ix == 0
        assert node_id.iy == 0
        
        # Node should be in graph
        assert node_id in graph._nodes
        
        # Get coordinates - should be snapped to grid center
        coords = graph.get_node_coordinates(node_id)
        assert coords == (0.5, 0.5)  # Grid center
        
        # Adding same logical point should return same node
        node_id2 = graph.add_node(0.1, 0.2)  # Same grid cell
        assert node_id2 == node_id
    
    def test_add_edge(self):
        """Test edge addition."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Create a simple linestring
        line = LineString([(0.1, 0.2), (1.8, 2.3), (3.9, 4.1)])
        attributes = {"highway": "primary", "name": "Test Road"}
        
        edge_key = graph.add_edge(line, 0, attributes)
        
        # Check that edge was added
        assert edge_key in graph._edges
        assert 0 in graph._edges[edge_key]
        assert edge_key in graph._edge_geometries
        assert graph._edge_attributes[edge_key] == attributes
        
        # Check that endpoints were created
        assert len(graph._nodes) == 2  # Start and end nodes
    
    def test_find_duplicates(self):
        """Test duplicate edge detection."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Add same edge twice (different feature indices)
        line1 = LineString([(0.1, 0.2), (1.1, 1.2)])
        line2 = LineString([(0.2, 0.1), (1.2, 1.1)])  # Slightly different but same after snapping
        
        edge_key1 = graph.add_edge(line1, 0)
        edge_key2 = graph.add_edge(line2, 1)
        
        # Should be same edge key (same snapped endpoints)
        assert edge_key1 == edge_key2
        
        # Find duplicates
        duplicates = graph.find_duplicates()
        assert len(duplicates) == 1
        assert edge_key1 in duplicates
        assert set(duplicates[edge_key1]) == {0, 1}
    
    def test_node_degree(self):
        """Test node degree calculation."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Create lines forming a T-junction
        line1 = LineString([(0, 0), (2, 0)])  # Horizontal
        line2 = LineString([(1, -1), (1, 1)])  # Vertical, intersects at (1,0)
        
        graph.add_edge(line1, 0)
        graph.add_edge(line2, 1)
        
        # Check degrees
        for node_id in graph._nodes:
            coords = graph.get_node_coordinates(node_id)
            if abs(coords[0] - 0.5) < 0.1 and abs(coords[1] - 0.5) < 0.1:  # Near (0,0)
                assert graph.get_node_degree(node_id) == 1
            elif abs(coords[0] - 2.5) < 0.1 and abs(coords[1] - 0.5) < 0.1:  # Near (2,0)
                assert graph.get_node_degree(node_id) == 1
            elif abs(coords[0] - 1.5) < 0.1 and abs(coords[1] - 0.5) < 0.1:  # Near (1,0) - intersection
                assert graph.get_node_degree(node_id) >= 2  # Should be intersection
    
    def test_snap_linestring(self):
        """Test linestring snapping."""
        config = CleanConfig(snap_precision=0.5)
        graph = QCKGraph(config, "EPSG:3857")
        
        original = LineString([(0.1, 0.2), (0.8, 1.3), (2.1, 2.9)])
        snapped = graph.snap_linestring(original)
        
        # Check that all coordinates are snapped
        coords = list(snapped.coords)
        for x, y in coords:
            # Should be at grid centers: (i + 0.5) * 0.5
            expected_x = (int(x / 0.5) + 0.5) * 0.5
            expected_y = (int(y / 0.5) + 0.5) * 0.5
            assert abs(x - expected_x) < 1e-10
            assert abs(y - expected_y) < 1e-10
    
    def test_export_nodes_gdf(self):
        """Test node export to GeoDataFrame."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Add some edges to create nodes
        line1 = LineString([(0, 0), (1, 1)])
        line2 = LineString([(1, 1), (2, 0)])
        
        graph.add_edge(line1, 0)
        graph.add_edge(line2, 1)
        
        # Export nodes
        nodes_gdf = graph.export_nodes_gdf()
        
        assert len(nodes_gdf) == 3  # Three unique nodes
        assert 'node_id' in nodes_gdf.columns
        assert 'ix' in nodes_gdf.columns
        assert 'iy' in nodes_gdf.columns
        assert 'degree' in nodes_gdf.columns
        assert 'geometry' in nodes_gdf.columns
        
        # Check CRS
        assert nodes_gdf.crs.to_string() == "EPSG:3857"
        
        # Check geometries are Points
        assert all(geom.geom_type == 'Point' for geom in nodes_gdf.geometry)
    
    def test_export_edges_gdf(self):
        """Test edge export to GeoDataFrame."""
        config = CleanConfig(snap_precision=1.0)
        graph = QCKGraph(config, "EPSG:3857")
        
        # Add an edge with attributes
        line = LineString([(0, 0), (1, 1)])
        attributes = {"highway": "primary"}
        graph.add_edge(line, 0, attributes)
        
        # Export edges
        edges_gdf = graph.export_edges_gdf()
        
        assert len(edges_gdf) == 1
        assert 'edge_key' in edges_gdf.columns
        assert 'node_a' in edges_gdf.columns
        assert 'node_b' in edges_gdf.columns
        assert 'feature_count' in edges_gdf.columns
        assert 'highway' in edges_gdf.columns  # Attribute should be preserved
        assert 'geometry' in edges_gdf.columns
        
        # Check that attribute was preserved
        assert edges_gdf.iloc[0]['highway'] == 'primary'
        
        # Check geometry type
        assert edges_gdf.iloc[0].geometry.geom_type == 'LineString'


class TestDatasetStats:
    """Tests for dataset statistics computation."""
    
    def test_compute_dataset_stats_simple(self):
        """Test dataset statistics with simple geometries."""
        # Create simple test data
        lines = [
            LineString([(0, 0), (1, 0), (2, 0)]),  # Length 2, vertex spacings [1, 1]
            LineString([(0, 1), (3, 1)]),  # Length 3, vertex spacing [3]
        ]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:4326')
        
        vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
        
        # Check edge lengths
        assert set(edge_lengths) == {2.0, 3.0}
        
        # Check vertex spacings
        expected_spacings = [1.0, 1.0, 3.0]  # From first line: 1, 1; from second line: 3
        assert sorted(vertex_spacings) == sorted(expected_spacings)
        
        # Check bbox diagonal
        # Bounds: [0, 0, 3, 1], diagonal = sqrt(3^2 + 1^2) = sqrt(10)
        expected_diagonal = np.sqrt(3**2 + 1**2)
        assert abs(bbox_diagonal - expected_diagonal) < 1e-10
    
    def test_compute_dataset_stats_multilinestring(self):
        """Test dataset statistics with MultiLineString."""
        multiline = MultiLineString([
            LineString([(0, 0), (1, 0)]),  # Length 1
            LineString([(2, 0), (2, 1)]),  # Length 1
        ])
        gdf = gpd.GeoDataFrame({'geometry': [multiline]}, crs='EPSG:4326')
        
        vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
        
        # Should have stats for both parts
        assert len(edge_lengths) == 2
        assert all(length == 1.0 for length in edge_lengths)
        assert len(vertex_spacings) == 2
        assert all(spacing == 1.0 for spacing in vertex_spacings)
    
    def test_compute_dataset_stats_empty(self):
        """Test dataset statistics with empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        
        vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
        
        assert len(vertex_spacings) == 0
        assert len(edge_lengths) == 0
        # bbox_diagonal should handle empty case gracefully
        assert not np.isfinite(bbox_diagonal) or bbox_diagonal == 0


class TestAnalyzeNetworkTopology:
    """Tests for network topology analysis."""
    
    def test_analyze_simple_network(self):
        """Test topology analysis on simple network."""
        lines = [
            LineString([(0, 0), (1, 0)]),  # Simple edge
            LineString([(1, 0), (2, 0)]),  # Connected edge
            LineString([(0, 1), (1, 1)]),  # Separate component
        ]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        analysis = analyze_network_topology(gdf, config, 'EPSG:3857')
        
        assert analysis['total_nodes'] == 4  # Four unique endpoints
        assert analysis['total_edges'] == 3  # Three edges
        assert analysis['duplicate_edge_keys'] == 0  # No duplicates
        assert analysis['total_duplicate_features'] == 0
        
        # Check node degrees
        degrees = analysis['node_degree_distribution']
        assert degrees['degree_1'] == 2  # Two endpoints
        assert degrees['degree_2'] == 2  # Two intermediate nodes
        assert degrees['degree_3_plus'] == 0
        
        assert analysis['max_node_degree'] == 2
    
    def test_analyze_network_with_duplicates(self):
        """Test topology analysis with duplicate edges."""
        lines = [
            LineString([(0, 0), (1, 0)]),  # Original
            LineString([(0.1, 0.1), (1.1, 0.1)]),  # Duplicate after snapping
            LineString([(2, 0), (3, 0)]),  # Different edge
        ]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        analysis = analyze_network_topology(gdf, config, 'EPSG:3857')
        
        assert analysis['total_edges'] == 2  # Two unique edges
        assert analysis['duplicate_edge_keys'] == 1  # One duplicate edge key
        assert analysis['total_duplicate_features'] == 2  # Two features share the key
    
    def test_analyze_network_with_multilinestring(self):
        """Test topology analysis with MultiLineString."""
        multiline = MultiLineString([
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
        ])
        gdf = gpd.GeoDataFrame({'geometry': [multiline]}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        analysis = analyze_network_topology(gdf, config, 'EPSG:3857')
        
        # MultiLineString should be treated as separate edges
        assert analysis['total_nodes'] == 3  # Three unique nodes: (0,0), (1,0), (2,0)
        assert analysis['total_edges'] == 2  # Two edges from the MultiLineString parts
    
    def test_analyze_network_junction(self):
        """Test topology analysis with junction (degree > 2)."""
        lines = [
            LineString([(0, 0), (1, 0)]),  # Horizontal to center
            LineString([(1, 0), (2, 0)]),  # Horizontal from center
            LineString([(1, -1), (1, 0)]),  # Vertical to center
            LineString([(1, 0), (1, 1)]),  # Vertical from center
        ]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs='EPSG:3857')
        config = CleanConfig(snap_precision=1.0)
        
        analysis = analyze_network_topology(gdf, config, 'EPSG:3857')
        
        # Should have one high-degree node at the junction
        degrees = analysis['node_degree_distribution']
        assert degrees['degree_1'] == 4  # Four endpoints
        assert degrees['degree_2'] == 0  # No degree-2 nodes
        assert degrees['degree_3_plus'] == 1  # One junction node
        
        assert analysis['max_node_degree'] == 4  # Junction has degree 4