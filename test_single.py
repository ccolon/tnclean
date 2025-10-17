#!/usr/bin/env python3
"""Test a single restructured pipeline test to verify it works."""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, 'src')

from tnclean import CleanConfig
from tnclean.modules import run_explode

def test_explode_restructured():
    """Test the restructured explode test to verify it works."""
    examples_dir = Path("examples")
    
    # Test config
    test_config = CleanConfig(
        metric_crs="auto",
        snap_precision=10.0,
        overlap_tolerance=5.0,
        simplify_tolerance=20.0,
        strict_validation=False,
        progress_bar=False
    )
    
    input_file = examples_dir / "explode_multilinestring.geojson"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.geojson"
        
        # Run only explode operation
        gdf, report = run_explode(str(input_file), str(output_file), test_config)
        
        # Verify input processing
        print(f"✅ Input count: {report['input_count']} (expected: 2)")
        assert report["input_count"] == 2, "Should start with 2 features"
        
        # Verify explosion occurred
        print(f"✅ Exploded count: {report['exploded_count']} (expected: >= 1)")
        assert report["exploded_count"] >= 1, "Should explode at least 1 MultiLineString"
        
        # Verify all geometries are LineString
        print(f"✅ All geometries are LineString: {gdf.geometry.geom_type.eq('LineString').all()}")
        assert gdf.geometry.geom_type.eq("LineString").all(), "All features should be LineString"
        
        # Verify attributes preserved
        highway_101_features = gdf[gdf["name"] == "Highway 101"]
        print(f"✅ Highway 101 segments: {len(highway_101_features)} (expected: 3)")
        assert len(highway_101_features) == 3, "Should have 3 Highway 101 segments"
        
        print(f"✅ All segments have lanes=4: {highway_101_features['lanes'].eq(4).all()}")
        assert highway_101_features["lanes"].eq(4).all(), "All segments should preserve lanes=4"
        
        print("🎉 test_explode_restructured PASSED!")

if __name__ == "__main__":
    test_explode_restructured()