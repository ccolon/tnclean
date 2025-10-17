"""Tests for core data types and configuration."""

import pytest
from tnclean.types import CleanConfig, NodeId, EdgeKey, ValidationBounds


class TestCleanConfig:
    """Tests for CleanConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CleanConfig()
        assert config.metric_crs == "auto"
        assert config.snap_precision == 1.0
        assert config.overlap_tolerance == 0.5
        assert config.simplify_tolerance == 2.0
        assert config.prune_epsilon == 0.1
        assert config.dedup_policy == "keep-shortest"
        assert config.keep_parallel_edges is False
        assert config.preserve_fields is None
        assert config.strict_validation is False
        assert config.progress_bar is True
        assert config.verbose == 0
    
    def test_config_validation_positive_tolerances(self):
        """Test that tolerances must be positive."""
        with pytest.raises(ValueError, match="snap_precision must be positive"):
            CleanConfig(snap_precision=0.0)
        
        with pytest.raises(ValueError, match="snap_precision must be positive"):
            CleanConfig(snap_precision=-1.0)
        
        with pytest.raises(ValueError, match="overlap_tolerance must be positive"):
            CleanConfig(overlap_tolerance=0.0)
        
        with pytest.raises(ValueError, match="overlap_tolerance must be positive"):
            CleanConfig(overlap_tolerance=-0.5)
    
    def test_config_validation_non_negative_tolerances(self):
        """Test that some tolerances can be zero but not negative."""
        config = CleanConfig(simplify_tolerance=0.0, prune_epsilon=0.0)
        assert config.simplify_tolerance == 0.0
        assert config.prune_epsilon == 0.0
        
        with pytest.raises(ValueError, match="simplify_tolerance must be non-negative"):
            CleanConfig(simplify_tolerance=-1.0)
        
        with pytest.raises(ValueError, match="prune_epsilon must be non-negative"):
            CleanConfig(prune_epsilon=-0.1)
    
    def test_config_validation_length_ratios(self):
        """Test length ratio bounds validation."""
        with pytest.raises(ValueError, match="Invalid pairing length ratio bounds"):
            CleanConfig(pairing_length_ratio_min=0.8, pairing_length_ratio_max=0.6)
        
        with pytest.raises(ValueError, match="Invalid pairing length ratio bounds"):
            CleanConfig(pairing_length_ratio_min=-0.1)
    
    def test_config_validation_overlap_alpha(self):
        """Test overlap alpha bounds validation."""
        with pytest.raises(ValueError, match="pairing_overlap_alpha must be in \\[0,1\\]"):
            CleanConfig(pairing_overlap_alpha=-0.1)
        
        with pytest.raises(ValueError, match="pairing_overlap_alpha must be in \\[0,1\\]"):
            CleanConfig(pairing_overlap_alpha=1.1)
        
        # Valid values
        config1 = CleanConfig(pairing_overlap_alpha=0.0)
        config2 = CleanConfig(pairing_overlap_alpha=1.0)
        assert config1.pairing_overlap_alpha == 0.0
        assert config2.pairing_overlap_alpha == 1.0


class TestNodeId:
    """Tests for NodeId (QCK) dataclass."""
    
    def test_node_id_creation(self):
        """Test NodeId creation and properties."""
        node = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        assert node.ix == 10
        assert node.iy == 20
        assert node.crs == "EPSG:3857"
        assert node.precision == 1.0
    
    def test_node_id_string_representation(self):
        """Test NodeId string conversion."""
        node = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        assert str(node) == "10:20:EPSG:3857:1.0"
    
    def test_node_id_hashable(self):
        """Test that NodeId is hashable and can be used in sets/dicts."""
        node1 = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        node2 = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        node3 = NodeId(ix=11, iy=20, crs="EPSG:3857", precision=1.0)
        
        # Equal nodes should have same hash
        assert hash(node1) == hash(node2)
        assert node1 == node2
        
        # Different nodes should be different
        assert node1 != node3
        
        # Can be used in sets
        node_set = {node1, node2, node3}
        assert len(node_set) == 2  # node1 and node2 are equal
    
    def test_node_id_frozen(self):
        """Test that NodeId is immutable."""
        node = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            node.ix = 15


class TestEdgeKey:
    """Tests for EdgeKey dataclass."""
    
    def test_edge_key_creation(self):
        """Test EdgeKey creation and canonical ordering."""
        node_a = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        node_b = NodeId(ix=30, iy=40, crs="EPSG:3857", precision=1.0)
        
        edge1 = EdgeKey(node_a, node_b)
        edge2 = EdgeKey(node_b, node_a)  # Reversed order
        
        # Should be canonically ordered
        assert edge1 == edge2
        assert hash(edge1) == hash(edge2)
    
    def test_edge_key_ordering(self):
        """Test that edge keys are ordered consistently."""
        node_small = NodeId(ix=5, iy=10, crs="EPSG:3857", precision=1.0)
        node_large = NodeId(ix=15, iy=20, crs="EPSG:3857", precision=1.0)
        
        edge = EdgeKey(node_large, node_small)  # Pass in reverse order
        
        # Should be reordered to canonical form
        assert edge.node_a.ix <= edge.node_b.ix or (edge.node_a.ix == edge.node_b.ix and edge.node_a.iy <= edge.node_b.iy)
    
    def test_edge_key_string_representation(self):
        """Test EdgeKey string conversion."""
        node_a = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        node_b = NodeId(ix=30, iy=40, crs="EPSG:3857", precision=1.0)
        edge = EdgeKey(node_a, node_b)
        
        edge_str = str(edge)
        assert " <-> " in edge_str
        assert str(node_a) in edge_str or str(node_b) in edge_str
    
    def test_edge_key_hashable(self):
        """Test that EdgeKey is hashable."""
        node_a = NodeId(ix=10, iy=20, crs="EPSG:3857", precision=1.0)
        node_b = NodeId(ix=30, iy=40, crs="EPSG:3857", precision=1.0)
        
        edge1 = EdgeKey(node_a, node_b)
        edge2 = EdgeKey(node_b, node_a)  # Same edge, different order
        
        edge_set = {edge1, edge2}
        assert len(edge_set) == 1  # Should deduplicate


class TestValidationBounds:
    """Tests for ValidationBounds dataclass."""
    
    def test_compute_bounds(self):
        """Test computation of validation bounds."""
        vertex_spacings = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
        edge_lengths = [10.0, 20.0, 50.0, 100.0, 200.0]
        bbox_diagonal = 1000.0
        
        bounds = ValidationBounds.compute_bounds(vertex_spacings, edge_lengths, bbox_diagonal)
        
        assert bounds.vertex_spacing_p05 == 0.1  # 5th percentile
        assert bounds.vertex_spacing_p50 == 1.25  # 50th percentile (median)
        assert bounds.median_edge_length == 50.0
        assert bounds.dataset_diagonal == 1000.0
        
        # Check derived bounds
        assert bounds.snap_precision_min == 0.25 * bounds.vertex_spacing_p05
        assert bounds.snap_precision_max == 0.75 * bounds.vertex_spacing_p50
        assert bounds.overlap_tolerance_max == 0.5
        assert bounds.simplify_tolerance_max == min(2 * bounds.median_edge_length, 0.05 * bbox_diagonal)
        assert bounds.prune_epsilon_max == 0.1 * bounds.vertex_spacing_p50
    
    def test_validate_config_pass(self):
        """Test config validation with valid parameters."""
        bounds = ValidationBounds(
            vertex_spacing_p05=0.1,
            vertex_spacing_p50=1.0,
            median_edge_length=50.0,
            dataset_diagonal=1000.0,
            snap_precision_min=0.025,
            snap_precision_max=0.75,
            overlap_tolerance_max=0.5,
            simplify_tolerance_max=100.0,
            prune_epsilon_max=0.1
        )
        
        config = CleanConfig(
            snap_precision=0.5,  # Within bounds
            overlap_tolerance=0.25,  # Within bounds
            simplify_tolerance=50.0,  # Within bounds
            prune_epsilon=0.05  # Within bounds
        )
        
        issues = bounds.validate_config(config)
        assert len(issues) == 0
    
    def test_validate_config_issues(self):
        """Test config validation with invalid parameters."""
        bounds = ValidationBounds(
            vertex_spacing_p05=0.1,
            vertex_spacing_p50=1.0,
            median_edge_length=50.0,
            dataset_diagonal=1000.0,
            snap_precision_min=0.025,
            snap_precision_max=0.75,
            overlap_tolerance_max=0.5,
            simplify_tolerance_max=100.0,
            prune_epsilon_max=0.1
        )
        
        config = CleanConfig(
            snap_precision=2.0,  # Too large
            overlap_tolerance=1.0,  # Too large relative to snap_precision
            simplify_tolerance=200.0,  # Too large
            prune_epsilon=0.5  # Too large
        )
        
        issues = bounds.validate_config(config)
        assert len(issues) > 0
        
        # Check that all expected issues are reported
        issue_text = " ".join(issues)
        assert "snap_precision" in issue_text
        assert "overlap_tolerance" in issue_text
        assert "simplify_tolerance" in issue_text
        assert "prune_epsilon" in issue_text
    
    def test_clamp_config(self):
        """Test config clamping to valid bounds."""
        bounds = ValidationBounds(
            vertex_spacing_p05=0.1,
            vertex_spacing_p50=1.0,
            median_edge_length=50.0,
            dataset_diagonal=1000.0,
            snap_precision_min=0.025,
            snap_precision_max=0.75,
            overlap_tolerance_max=0.5,
            simplify_tolerance_max=100.0,
            prune_epsilon_max=0.1
        )
        
        config = CleanConfig(
            snap_precision=2.0,  # Will be clamped to 0.75
            overlap_tolerance=2.0,  # Will be clamped
            simplify_tolerance=200.0,  # Will be clamped to 100.0
            prune_epsilon=0.5  # Will be clamped
        )
        
        clamped = bounds.clamp_config(config)
        
        assert clamped.snap_precision == 0.75
        assert clamped.overlap_tolerance <= 0.5 * clamped.snap_precision
        assert clamped.simplify_tolerance == 100.0
        assert clamped.prune_epsilon <= 0.1
        
        # Original config should be unchanged
        assert config.snap_precision == 2.0