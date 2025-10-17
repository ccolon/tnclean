"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from tnclean.types import CleanConfig


@pytest.fixture
def simple_linestring():
    """Simple LineString for testing."""
    return LineString([(0, 0), (1, 0), (2, 0)])


@pytest.fixture
def simple_multilinestring():
    """Simple MultiLineString for testing."""
    return MultiLineString([
        LineString([(0, 0), (1, 0)]),
        LineString([(2, 0), (3, 0)]),
    ])


@pytest.fixture
def simple_gdf():
    """Simple GeoDataFrame for testing."""
    lines = [
        LineString([(0, 0), (1, 0)]),
        LineString([(1, 0), (2, 0)]),
        LineString([(0, 1), (1, 1)]),
    ]
    return gpd.GeoDataFrame({
        'geometry': lines,
        'highway': ['primary', 'secondary', 'tertiary'],
        'name': ['Main St', 'Side St', 'Back St'],
    }, crs='EPSG:4326')


@pytest.fixture
def multiline_gdf():
    """GeoDataFrame with MultiLineStrings for testing."""
    multiline = MultiLineString([
        LineString([(0, 0), (1, 0)]),
        LineString([(2, 0), (3, 0)]),
    ])
    line = LineString([(0, 1), (1, 1)])
    
    return gpd.GeoDataFrame({
        'geometry': [multiline, line],
        'highway': ['trunk', 'primary'],
    }, crs='EPSG:4326')


@pytest.fixture
def default_config():
    """Default CleanConfig for testing."""
    return CleanConfig(
        progress_bar=False,  # Disable progress bars in tests
        verbose=0,  # Quiet logging in tests
    )


@pytest.fixture
def projected_gdf():
    """GeoDataFrame in projected coordinates for testing."""
    lines = [
        LineString([(100000, 200000), (101000, 200000)]),  # 1km line
        LineString([(101000, 200000), (102000, 200000)]),  # Connected 1km line
    ]
    return gpd.GeoDataFrame({
        'geometry': lines,
        'highway': ['primary', 'secondary'],
    }, crs='EPSG:3857')


# Hypothesis strategies for property-based testing
coordinate_strategy = st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)

linestring_coordinates = st.lists(
    st.tuples(coordinate_strategy, coordinate_strategy),
    min_size=2,
    max_size=10
).map(lambda coords: LineString(coords) if len(set(coords)) >= 2 else LineString([(0, 0), (1, 1)]))

@st.composite
def valid_linestring(draw):
    """Strategy for generating valid LineStrings."""
    coords = draw(st.lists(
        st.tuples(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
        ),
        min_size=2,
        max_size=20
    ))
    
    # Ensure coordinates are not all identical
    if len(set(coords)) < 2:
        coords = [(0, 0), (1, 1)]
    
    return LineString(coords)


@st.composite 
def valid_multilinestring(draw):
    """Strategy for generating valid MultiLineStrings."""
    num_parts = draw(st.integers(min_value=1, max_value=5))
    parts = []
    
    for i in range(num_parts):
        linestring = draw(valid_linestring())
        parts.append(linestring)
    
    return MultiLineString(parts)


@st.composite
def clean_config_strategy(draw):
    """Strategy for generating valid CleanConfig instances."""
    snap_precision = draw(st.floats(min_value=0.01, max_value=100.0))
    overlap_tolerance = draw(st.floats(min_value=0.01, max_value=snap_precision * 0.5))
    simplify_tolerance = draw(st.floats(min_value=snap_precision, max_value=snap_precision * 10))
    prune_epsilon = draw(st.floats(min_value=0.0, max_value=min(overlap_tolerance, simplify_tolerance * 0.5)))
    
    return CleanConfig(
        snap_precision=snap_precision,
        overlap_tolerance=overlap_tolerance,
        simplify_tolerance=simplify_tolerance,
        prune_epsilon=prune_epsilon,
        dedup_policy=draw(st.sampled_from(['keep-shortest', 'keep-first', 'keep-longest'])),
        keep_parallel_edges=draw(st.booleans()),
        progress_bar=False,
        verbose=0,
    )