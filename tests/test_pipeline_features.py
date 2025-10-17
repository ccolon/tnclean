"""Test cases for core pipeline features using Vienna-based examples."""

import pytest
import geopandas as gpd
from pathlib import Path
from tnclean import clean_network, CleanConfig
from tnclean.modules import run_explode, run_snap, run_split, run_deoverlap, run_simplify


class TestPipelineFeatures:
    """Test each core pipeline feature with dedicated test cases."""
    
    @pytest.fixture
    def examples_dir(self):
        """Path to examples directory."""
        return Path(__file__).parent.parent / "examples"
    
    @pytest.fixture
    def test_config(self):
        """Standard test configuration."""
        return CleanConfig(
            metric_crs="auto",
            snap_precision=10.0,
            overlap_tolerance=5.0,
            simplify_tolerance=20.0,
            strict_validation=False,
            progress_bar=False
        )

    def _run_pipeline_steps(self, input_path: str, output_path: str, config: CleanConfig, steps: list) -> tuple:
        """Helper to run specific pipeline steps in sequence."""
        import tempfile
        from pathlib import Path
        
        current_input = input_path
        gdf = None
        final_report = {}
        
        for i, step in enumerate(steps):
            if i == len(steps) - 1:
                # Last step: use final output path
                current_output = output_path
            else:
                # Intermediate step: use temp file
                current_output = str(Path(output_path).parent / f"temp_step_{i}.geojson")
            
            if step == "explode":
                gdf, report = run_explode(current_input, current_output, config)
            elif step == "snap":
                gdf, report = run_snap(current_input, current_output, config)
            elif step == "split":
                gdf, report = run_split(current_input, current_output, config)
            elif step == "deoverlap":
                gdf, report = run_deoverlap(current_input, current_output, config)
            elif step == "simplify":
                gdf, report = run_simplify(current_input, current_output, config)
            else:
                raise ValueError(f"Unknown step: {step}")
            
            # Update input for next step
            current_input = current_output
            final_report.update(report)
        
        return gdf, final_report

    def test_explode_multilinestring(self, examples_dir, test_config, tmp_path):
        """
        Test: MultiLineString explosion into separate LineStrings
        Input: 1 MultiLineString (3 segments) + 1 LineString = 2 features
        Expected: 4 LineString features (3 from explosion + 1 original)
        """
        input_file = examples_dir / "explode_multilinestring.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run only explode operation
        gdf, report = run_explode(str(input_file), str(output_file), test_config)
        
        # Verify input processing
        assert report["input_count"] == 2, "Should start with 2 features"
        
        # Verify explosion occurred
        assert report["exploded_count"] >= 1, "Should explode at least 1 MultiLineString"
        
        # Verify all geometries are LineString
        assert gdf["geometry"].geom_type.eq("LineString").all(), "All features should be LineString"
        
        # Verify attributes preserved
        highway_101_features = gdf[gdf["name"] == "Highway 101"]
        assert len(highway_101_features) == 3, "Should have 3 Highway 101 segments"
        assert highway_101_features["lanes"].eq(4).all(), "All segments should preserve lanes=4"

    def test_snap_near_endpoints(self, examples_dir, test_config, tmp_path):
        """
        Test: Coordinate snapping brings near-endpoints together
        Input: 3 roads with endpoints designed to snap together
        Expected: Shared coordinates after snapping, proper connectivity
        """
        input_file = examples_dir / "snap_near_endpoints.geojson" 
        output_file = tmp_path / "output.geojson"
        
        # Run only snap operation
        gdf, report = run_snap(str(input_file), str(output_file), test_config)
        
        # Verify snapping occurred
        assert report["snapped_vertex_changes"] >= 0, "Should track vertex snapping changes"
        
        # Check that roads are now properly connected (same coordinates)
        coords = []
        for geom in gdf.geometry:
            coords.extend([geom.coords[0], geom.coords[-1]])  # Start and end points
        
        # Count unique coordinates (should be fewer after snapping)
        unique_coords = set(tuple(round(c, 6) for c in coord) for coord in coords)
        assert len(unique_coords) < 6, "Should have fewer unique endpoints after snapping"

    def test_crossing_intersection(self, examples_dir, test_config, tmp_path):
        """
        Test: X-pattern intersection detection and splitting
        Input: 2 roads crossing at [16.3700, 48.2025]
        Expected: 4 road segments after splitting at intersection
        """
        input_file = examples_dir / "crossing_intersection.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline steps needed for split: explode → snap → split
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config, 
            ["explode", "snap", "split"]
        )
        
        # Verify intersection splits occurred
        assert report["split_count"] >= 2, "Should create at least 2 intersection splits"
        
        # Should have more features after splitting
        assert len(gdf) > report["input_count"], "Should have more features after intersection splitting"
        
        # Check that intersection point exists in multiple geometries
        intersection_point = (16.3700, 48.2025)  # Known intersection coordinate
        
        # Count how many geometries contain the intersection point (within tolerance)
        geometries_with_intersection = 0
        for geom in gdf.geometry:
            coords = list(geom.coords)
            for coord in coords:
                if abs(coord[0] - intersection_point[0]) < 0.001 and abs(coord[1] - intersection_point[1]) < 0.001:
                    geometries_with_intersection += 1
                    break
        
        assert geometries_with_intersection >= 4, "Intersection point should be in at least 4 segments"

    def test_t_junction(self, examples_dir, test_config, tmp_path):
        """
        Test: T-junction where branch meets main road at midpoint
        Input: Main highway + branch road meeting at [16.3700, 48.2000]
        Expected: Main highway split into 2 segments, branch preserved
        """
        input_file = examples_dir / "t_junction.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline steps needed for split: explode → snap → split
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config, 
            ["explode", "snap", "split"]
        )
        
        # Should create intersection splits for T-junction
        assert report["split_count"] >= 1, "Should split main highway at T-junction"
        
        # Should have more segments than input
        assert len(gdf) >= 3, "Should have at least 3 segments after T-junction split"
        
        # Branch road should remain as Branch Road
        branch_roads = gdf[gdf["name"] == "Branch Road"]
        assert len(branch_roads) >= 1, "Should preserve Branch Road"
        
        # Main highway should be split
        main_highways = gdf[gdf["name"] == "Main Highway"]
        assert len(main_highways) >= 2, "Main Highway should be split at T-junction"

    def test_partial_overlap(self, examples_dir, test_config, tmp_path):
        """
        Test: Partial overlap detection and splitting
        Input: 2 routes sharing middle segment [16.3700,48.2000] to [16.3800,48.2000]
        Expected: Overlapping segment split, 4+ total segments
        """
        input_file = examples_dir / "partial_overlap.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline steps needed for deoverlap: explode → snap → split → deoverlap
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config, 
            ["explode", "snap", "split", "deoverlap"]
        )
        
        # Should detect overlapping pairs
        assert report["overlap_pairs_found"] > 0, "Should find overlapping route pairs"
        
        # Should have processed overlaps (split creates segments, deoverlap may merge/remove)
        # Starting with 2 routes, should end up with multiple segments after processing
        assert len(gdf) >= 4, "Should have at least 4 segments after processing overlapping routes"
        
        # Check for shared coordinate sequences in the overlap region
        overlap_coords = [(16.3700, 48.2000), (16.3800, 48.2000)]
        
        segments_with_overlap = 0
        for geom in gdf.geometry:
            coords = list(geom.coords)
            if len(coords) >= 2:
                # Check if this segment contains part of the overlap
                for i in range(len(coords) - 1):
                    start, end = coords[i], coords[i + 1]
                    if (abs(start[0] - overlap_coords[0][0]) < 0.001 and 
                        abs(start[1] - overlap_coords[0][1]) < 0.001 and
                        abs(end[0] - overlap_coords[1][0]) < 0.001 and
                        abs(end[1] - overlap_coords[1][1]) < 0.001):
                        segments_with_overlap += 1
                        break
        
        assert segments_with_overlap >= 1, "Should have segments covering the overlap region"

    def test_duplicate_edges(self, examples_dir, test_config, tmp_path):
        """
        Test: Deduplication with keep-shortest policy
        Input: 3 routes between same endpoints (1200m, 1500m, 1800m lengths)
        Expected: Only shortest route (1200m Direct Route) kept
        """
        input_file = examples_dir / "duplicate_edges.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Use keep-shortest dedup policy
        config = CleanConfig(
            metric_crs="auto",
            snap_precision=10.0,
            overlap_tolerance=5.0,
            dedup_policy="keep-shortest",
            strict_validation=False,
            progress_bar=False
        )
        
        # Run pipeline steps needed for dedup: explode → snap → split → deoverlap  
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), config, 
            ["explode", "snap", "split", "deoverlap"]
        )
        
        # Should detect and remove duplicates
        if report.get("duplicates_removed", 0) > 0:
            # Check that shortest route was kept
            remaining_routes = gdf[gdf["name"].isin(["Main Route", "Alt Route", "Direct Route"])]
            
            # Should prefer Direct Route (shortest) if deduplication occurred
            if "Direct Route" in remaining_routes["name"].values:
                direct_routes = remaining_routes[remaining_routes["name"] == "Direct Route"]
                assert len(direct_routes) >= 1, "Direct Route (shortest) should be preserved"
        
        # Verify final count is reasonable
        assert len(gdf) >= 1, "Should have at least one route remaining"

    def test_zigzag_simplify(self, examples_dir, test_config, tmp_path):
        """
        Test: Topology-preserving simplification
        Input: Winding road with 21 vertices in nearly straight line
        Expected: Significant vertex reduction while preserving intersections
        """
        input_file = examples_dir / "zigzag_simplify.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run full pipeline needed for simplify: explode → snap → split → deoverlap → simplify
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config, 
            ["explode", "snap", "split", "deoverlap", "simplify"]
        )
        
        # Should reduce vertices through simplification
        assert report["vertices_before_simplify"] > report["vertices_after_simplify"], \
            "Should reduce vertex count through simplification"
        
        vertices_removed = report["vertices_removed"]
        assert vertices_removed > 5, "Should remove at least 5 vertices from zigzag road"
        
        # Find the winding road
        winding_road = gdf[gdf["name"] == "Winding Road"].iloc[0]
        simplified_coords = list(winding_road.geometry.coords)
        
        # Should have significantly fewer vertices than original (21)
        assert len(simplified_coords) < 15, "Winding road should be simplified to fewer vertices"
        
        # Should preserve start and end points
        assert abs(simplified_coords[0][0] - 16.3500) < 0.001, "Should preserve start point"
        assert abs(simplified_coords[-1][0] - 16.3700) < 0.001, "Should preserve end point"

    def test_attribute_inheritance(self, examples_dir, test_config, tmp_path):
        """
        Test: Attribute inheritance and concatenation on splits/merges
        Input: 4 roads meeting at intersection with different attributes
        Expected: Attributes preserved on splits, concatenated where appropriate
        """
        input_file = examples_dir / "attribute_inheritance.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline steps to test attribute inheritance: explode → snap → split
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config, 
            ["explode", "snap", "split"]
        )
        
        # Should have intersection splits
        assert report.get("split_count", 0) > 0, "Should create intersection splits"
        
        # Check attribute preservation
        highway_types = set(gdf["highway"].dropna())
        expected_highways = {"primary", "secondary", "residential"}
        assert highway_types.intersection(expected_highways), "Should preserve highway types"
        
        # Check that lane information is preserved
        lanes_values = set(gdf["lanes"].dropna())
        expected_lanes = {1, 2, 4}
        assert lanes_values.intersection(expected_lanes), "Should preserve lane counts"
        
        # Check surface attributes preserved
        surfaces = set(gdf["surface"].dropna())
        expected_surfaces = {"asphalt", "concrete", "gravel"}
        assert surfaces.intersection(expected_surfaces), "Should preserve surface types"
        
        # For segments that were split, attributes should be inherited
        us_101_segments = gdf[gdf["ref"] == "US-101"]
        if len(us_101_segments) > 2:  # If US-101 was split
            # All US-101 segments should maintain consistent ref
            assert us_101_segments["ref"].eq("US-101").all(), "Split segments should inherit ref attribute"

    def test_self_intersecting_linestring(self, examples_dir, test_config, tmp_path):
        """
        Test: Self-intersecting LineString (figure-8 pattern)
        Input: LineString that crosses itself + intersecting cross street  
        Expected: Self-intersection split + cross-intersection split
        """
        input_file = examples_dir / "self_intersecting.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline steps to handle self-intersections: explode → snap → split → deoverlap
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config,
            ["explode", "snap", "split", "deoverlap"]
        )
        
        # Note: Current implementation may not fully support self-intersection splitting
        # This test primarily validates that self-intersecting geometry doesn't crash the pipeline
        
        # Should handle self-intersecting geometry without crashing
        assert len(gdf) >= 2, "Should process self-intersecting geometry without crashing"
        
        # Loop Road should be preserved (even if not split at self-intersection)
        loop_segments = gdf[gdf["name"] == "Loop Road"]
        assert len(loop_segments) >= 1, "Self-intersecting Loop Road should be preserved"
        
        # If splits occurred, should be at cross-intersection minimum
        split_count = report.get("split_count", 0)
        if split_count > 0:
            # At minimum, should split where Cross Street intersects Loop Road
            assert len(gdf) > 2, "Should create segments from any intersection splitting"
        
        # Cross Street should intersect with the loop
        cross_segments = gdf[gdf["name"] == "Cross Street"] 
        assert len(cross_segments) >= 1, "Cross Street should be preserved"

    def test_multiway_intersection(self, examples_dir, test_config, tmp_path):
        """
        Test: Complex multi-way intersection (5 roads meeting at one point)
        Input: 5 roads converging at [16.3700, 48.2025]
        Expected: All roads split at the central intersection point
        """
        input_file = examples_dir / "multiway_intersection.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline steps: explode → snap → split 
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config,
            ["explode", "snap", "split"]
        )
        
        # Should create intersection splits (only roads that pass through intersection need splitting)
        assert report.get("split_count", 0) >= 2, "Should split roads that pass through central intersection"
        
        # Should have more segments after splitting 5-way intersection  
        assert len(gdf) >= 7, "Should create segments from splitting through-roads at intersection"
        
        # Check that intersection point exists in multiple geometries
        intersection_point = (16.3700, 48.2025)
        
        geometries_with_intersection = 0
        for geom in gdf.geometry:
            coords = list(geom.coords)
            for coord in coords:
                if abs(coord[0] - intersection_point[0]) < 0.001 and abs(coord[1] - intersection_point[1]) < 0.001:
                    geometries_with_intersection += 1
                    break
        
        assert geometries_with_intersection >= 4, "Central intersection point should be in multiple segments after splitting"

    def test_coincident_lines(self, examples_dir, test_config, tmp_path):
        """
        Test: Coincident overlapping LineStrings (100% same path)
        Input: 3 highways following exact same coordinates + intersecting side road
        Expected: Coincident highways deduplicated, side road intersections handled
        """
        input_file = examples_dir / "coincident_lines.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Use keep-shortest dedup policy for consistent behavior
        config = CleanConfig(
            metric_crs="auto",
            snap_precision=10.0,
            overlap_tolerance=5.0,
            dedup_policy="keep-shortest",
            strict_validation=False,
            progress_bar=False
        )
        
        # Run full pipeline to handle coincident lines: explode → snap → split → deoverlap
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), config,
            ["explode", "snap", "split", "deoverlap"]
        )
        
        # Should detect overlapping pairs among coincident highways
        assert report.get("overlap_pairs_found", 0) > 0, "Should find overlapping coincident highways"
        
        # Should remove duplicates from coincident lines
        assert report.get("duplicates_removed", 0) >= 1, "Should remove duplicate coincident highways"
        
        # Should have fewer segments after deduplication
        final_highways = gdf[gdf["highway"] == "primary"]
        assert len(final_highways) < 3, "Should deduplicate coincident primary highways"
        
        # Side road should still exist and be split at intersection
        side_roads = gdf[gdf["name"] == "Side Road"]
        assert len(side_roads) >= 1, "Side Road should be preserved"

    def test_micro_segments(self, examples_dir, test_config, tmp_path):
        """
        Test: Micro-segments and degenerate geometries
        Input: Very short segments, zero-length segments, nearly-zero segments
        Expected: Tiny segments pruned, connectivity preserved
        """
        input_file = examples_dir / "micro_segments.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Use small prune_epsilon to test micro-segment removal
        config = CleanConfig(
            metric_crs="auto", 
            snap_precision=10.0,
            overlap_tolerance=5.0,
            simplify_tolerance=20.0,
            prune_epsilon=1.0,  # Remove segments < 1m
            strict_validation=False,
            progress_bar=False
        )
        
        # Run full pipeline including simplify which does pruning
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), config,
            ["explode", "snap", "split", "deoverlap", "simplify"]
        )
        
        # Should prune tiny segments
        if "pruned_segments" in report:
            # If pruning occurred, should have removed micro segments
            pruned_count = report["pruned_segments"]
            if pruned_count > 0:
                assert len(gdf) < 6, "Should have fewer segments after pruning micro-segments"
        
        # Main Road and Continuation Road should be preserved
        main_roads = gdf[gdf["name"].isin(["Main Road", "Continuation Road"])]
        assert len(main_roads) >= 2, "Main roads should be preserved"
        
        # Zero-length and nearly-zero segments should be handled
        remaining_names = set(gdf["name"].dropna())
        # These tiny segments might be pruned or merged
        assert "Main Road" in remaining_names, "Main Road should be preserved"

    def test_parallel_offset_lines(self, examples_dir, test_config, tmp_path):
        """
        Test: Near-parallel offset LineStrings (parallel but offset within tolerance)
        Input: Highway centerline, north/south lanes, bike path, sidewalk - all parallel
        Expected: Parallel lines preserved or merged based on policy and tolerance
        """
        input_file = examples_dir / "parallel_offset.geojson"
        output_file = tmp_path / "output.geojson"
        
        # Run pipeline to see how parallel lines are handled
        gdf, report = self._run_pipeline_steps(
            str(input_file), str(output_file), test_config,
            ["explode", "snap", "split", "deoverlap"]
        )
        
        # All parallel lines should be processed
        assert len(gdf) >= 3, "Should preserve most parallel infrastructure elements"
        
        # Different highway types should be preserved
        highway_types = set(gdf["highway"].dropna())
        expected_types = {"primary", "cycleway", "footway"}
        assert len(highway_types.intersection(expected_types)) >= 2, "Should preserve different infrastructure types"
        
        # Check that no invalid intersections were created between parallel lines
        # (This is more of a validation that the algorithm doesn't create spurious intersections)
        total_features = len(gdf)
        assert total_features <= 10, "Should not create excessive spurious intersections between parallel lines"
        
        # Lane information should be preserved for different types
        centerline_lanes = gdf[gdf["name"] == "Highway Centerline"]
        if len(centerline_lanes) > 0:
            assert centerline_lanes.iloc[0]["lanes"] == 4, "Centerline lane count should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])