"""Core geometric operations for the tnclean pipeline."""

import logging
import warnings
from typing import List, Dict, Any, Tuple, Optional, Iterator
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely import get_coordinates, STRtree
from shapely.ops import linemerge
import pyproj
from pyproj import CRS, Transformer

from .types import CleanConfig, ProcessingStats, ValidationBounds
from .graph import QCKGraph, compute_dataset_stats

logger = logging.getLogger(__name__)


def auto_select_crs(gdf: gpd.GeoDataFrame) -> str:
    """Automatically select an appropriate metric CRS based on dataset extent.
    
    Selection logic:
    - Dataset diagonal ≤ 800km: Select UTM zone from centroid  
    - |latitude| ≥ 84°: Use Universal Polar Stereographic
    - Otherwise: Custom Lambert Azimuthal Equal-Area
    """
    # Get dataset bounds in geographic coordinates
    if not gdf.crs.is_geographic:
        gdf_geo = gdf.to_crs('EPSG:4326')
    else:
        gdf_geo = gdf
    
    bounds = gdf_geo.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Compute approximate diagonal in meters using great circle distance
    from geopy.distance import great_circle
    diagonal_km = great_circle(
        (bounds[1], bounds[0]),  # SW corner
        (bounds[3], bounds[2])   # NE corner
    ).kilometers
    
    # Decision logic
    if abs(center_lat) >= 84:
        # Polar regions: use Universal Polar Stereographic
        if center_lat > 0:
            return "EPSG:5041"  # UPS North
        else:
            return "EPSG:5042"  # UPS South
    
    elif diagonal_km <= 800:
        # Small datasets: use UTM
        utm_zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            epsg_code = 32600 + utm_zone  # UTM North
        else:
            epsg_code = 32700 + utm_zone  # UTM South
        return f"EPSG:{epsg_code}"
    
    else:
        # Large datasets: custom Lambert Azimuthal Equal-Area
        proj_string = (
            f"+proj=laea +lat_0={center_lat:.6f} +lon_0={center_lon:.6f} "
            f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        return proj_string


def explode_multilinestrings(
    gdf: gpd.GeoDataFrame, 
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Explode MultiLineStrings into individual LineStrings.
    
    Preserves all attributes for each exploded part.
    """
    logger.info("Exploding MultiLineStrings to LineStrings")
    
    exploded_rows = []
    multilinestring_count = 0
    
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        attributes = dict(row.drop('geometry'))
        
        if geometry.geom_type == 'MultiLineString':
            multilinestring_count += 1
            
            # Explode each part with inherited attributes
            for i, linestring in enumerate(geometry.geoms):
                if linestring.geom_type == 'LineString':
                    exploded_rows.append({
                        'geometry': linestring,
                        **attributes,
                        '_original_idx': idx,
                        '_part_idx': i
                    })
                    stats.exploded_count += 1
        
        elif geometry.geom_type == 'LineString':
            exploded_rows.append({
                'geometry': geometry,
                **attributes,
                '_original_idx': idx,
                '_part_idx': 0
            })
        else:
            logger.warning(f"Skipping unsupported geometry type: {geometry.geom_type}")
    
    stats.input_multilinestring_count = multilinestring_count
    logger.info(f"Exploded {multilinestring_count} MultiLineStrings into {stats.exploded_count} LineStrings")
    
    return gpd.GeoDataFrame(exploded_rows, crs=gdf.crs)


def snap_coordinates(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig,
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Snap coordinates to metric grid for topology stability."""
    logger.info(f"Snapping coordinates to {config.snap_precision}m grid")
    
    snapped_rows = []
    vertex_changes = 0
    
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        attributes = dict(row.drop('geometry'))
        
        if geometry.geom_type == 'LineString':
            coords = get_coordinates(geometry)
            snapped_coords = []
            
            for x, y in coords:
                # Quantize to grid
                precision = config.snap_precision
                ix = int(np.floor(x / precision))
                iy = int(np.floor(y / precision))
                
                # Snap to grid center
                snapped_x = (ix + 0.5) * precision
                snapped_y = (iy + 0.5) * precision
                
                # Count changes
                if abs(x - snapped_x) > 1e-10 or abs(y - snapped_y) > 1e-10:
                    vertex_changes += 1
                
                snapped_coords.append((snapped_x, snapped_y))
            
            snapped_geometry = LineString(snapped_coords)
            snapped_rows.append({'geometry': snapped_geometry, **attributes})
        
        else:
            logger.warning(f"Skipping non-LineString geometry in snap operation")
            snapped_rows.append({'geometry': geometry, **attributes})
    
    stats.snapped_vertex_changes = vertex_changes
    logger.info(f"Snapped {vertex_changes} vertices")
    
    return gpd.GeoDataFrame(snapped_rows, crs=gdf.crs)


def validate_and_clamp_config(
    gdf: gpd.GeoDataFrame, 
    config: CleanConfig
) -> Tuple[CleanConfig, List[str]]:
    """Validate configuration against dataset statistics and clamp if needed."""
    logger.info("Computing dataset statistics for tolerance validation")
    
    # Compute dataset statistics
    vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
    
    if not vertex_spacings or not edge_lengths:
        logger.warning("Unable to compute dataset statistics - using original config")
        return config, []
    
    # Compute validation bounds
    bounds = ValidationBounds.compute_bounds(vertex_spacings, edge_lengths, bbox_diagonal)
    
    # Validate configuration
    issues = bounds.validate_config(config, strict=config.strict_validation)
    
    if issues and config.strict_validation:
        # In strict mode, raise error
        error_msg = "Configuration validation failed in strict mode:\n" + "\n".join(f"  - {issue}" for issue in issues)
        raise ValueError(error_msg)
    
    elif issues:
        # In non-strict mode, clamp and warn
        logger.warning("Configuration issues found - auto-clamping:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Always clamp config to handle "auto" values, regardless of validation issues
    clamped_config = bounds.clamp_config(config)
    
    if issues:
        logger.info("Clamped configuration applied with issue corrections")
    else:
        logger.info("Configuration clamped (auto values converted)")
    
    return clamped_config, issues


def setup_crs_transform(gdf: gpd.GeoDataFrame, config: CleanConfig) -> Tuple[str, str, Optional[Transformer], Optional[Transformer]]:
    """Set up coordinate reference system transformation."""
    original_crs = gdf.crs.to_string()
    
    # Determine target metric CRS
    if config.metric_crs == "auto":
        target_crs = auto_select_crs(gdf)
        logger.info(f"Auto-selected metric CRS: {target_crs}")
    else:
        target_crs = config.metric_crs
    
    # Set up transformers if needed
    to_metric = None
    to_original = None
    
    if original_crs != target_crs:
        to_metric = Transformer.from_crs(original_crs, target_crs, always_xy=True)
        to_original = Transformer.from_crs(target_crs, original_crs, always_xy=True)
        logger.info(f"Set up CRS transformation: {original_crs} <-> {target_crs}")
    else:
        logger.info("No CRS transformation needed")
    
    return original_crs, target_crs, to_metric, to_original


def transform_to_metric(gdf: gpd.GeoDataFrame, to_metric: Optional[Transformer], target_crs: str) -> gpd.GeoDataFrame:
    """Transform GeoDataFrame to metric CRS."""
    if to_metric is None:
        return gdf
    
    logger.info("Transforming to metric CRS")
    return gdf.to_crs(target_crs)


def transform_to_original(gdf: gpd.GeoDataFrame, to_original: Optional[Transformer], original_crs: str) -> gpd.GeoDataFrame:
    """Transform GeoDataFrame back to original CRS."""
    if to_original is None:
        return gdf
    
    logger.info("Transforming back to original CRS")
    return gdf.to_crs(original_crs)


def split_at_intersections(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig,
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Split LineStrings at intersection points.
    
    Uses STRtree spatial indexing to find potential intersections, then computes
    exact intersections with Shapely. Handles crossing intersections, T-junctions,
    and self-intersections with tolerance-based near-miss detection.
    """
    logger.info("Splitting LineStrings at intersections")
    
    if len(gdf) < 2:
        stats.split_count = 0
        logger.info("Created 0 splits at intersections")
        return gdf.copy()
    
    total_features = len(gdf)
    logger.info(f"Processing {total_features} features for intersections...")
    
    # Build spatial index for efficient intersection queries
    logger.info("Building spatial index for intersection detection...")
    sindex = gdf.sindex
    tolerance = config.overlap_tolerance
    split_count = 0
    
    # Store all intersection points grouped by original geometry index
    intersections_by_geom = {}  # {geom_idx: [intersection_points]}
    
    # Phase 1: Find all intersection points
    logger.info("Phase 1/2: Finding intersection points...")
    intersection_pairs_checked = 0
    for idx in range(len(gdf)):
        if idx > 0 and idx % max(1, total_features // 10) == 0:
            progress_pct = (idx / total_features) * 100
            logger.info(f"  Progress: {idx}/{total_features} features processed ({progress_pct:.0f}%)")
        geom = gdf.iloc[idx].geometry
        
        # Find potential intersecting geometries using spatial index
        possible_matches_idx = list(sindex.intersection(geom.bounds))
        logger.debug(f"Geometry {idx}: found {len(possible_matches_idx)} potential intersections")
        
        if idx not in intersections_by_geom:
            intersections_by_geom[idx] = []
        
        for other_idx in possible_matches_idx:
            # Skip self (but handle self-intersections separately)
            if idx == other_idx:
                # Check for self-intersections
                if not geom.is_simple:
                    self_intersections = _find_self_intersections(geom, tolerance)
                    intersections_by_geom[idx].extend(self_intersections)
                continue
            
            # Only process each pair once (avoid duplicate work)
            if other_idx > idx:
                other_geom = gdf.iloc[other_idx].geometry
                
                # Find intersection points between the two geometries
                intersection_points = _find_intersection_points(geom, other_geom, tolerance)
                logger.debug(f"Intersection between geom {idx} and {other_idx}: found {len(intersection_points)} points")
                for i, pt in enumerate(intersection_points):
                    logger.debug(f"  Point {i}: {pt}")
                
                # Add intersection points to both geometries
                intersections_by_geom[idx].extend(intersection_points)
                if other_idx not in intersections_by_geom:
                    intersections_by_geom[other_idx] = []
                intersections_by_geom[other_idx].extend(intersection_points)
    
    # Summary of Phase 1
    total_intersection_points = sum(len(points) for points in intersections_by_geom.values())
    geometries_with_intersections = len([g for g in intersections_by_geom.values() if g])
    logger.info(f"Phase 1 complete: Found {total_intersection_points} intersection points affecting {geometries_with_intersections} geometries")
    
    # Phase 2: Split geometries at intersection points
    logger.info("Phase 2/2: Splitting geometries at intersection points...")
    split_rows = []
    
    for i, (idx, row) in enumerate(gdf.iterrows()):
        if i > 0 and i % max(1, total_features // 10) == 0:
            progress_pct = (i / total_features) * 100
            logger.info(f"  Progress: {i}/{total_features} geometries split ({progress_pct:.0f}%)")
        geom = row.geometry
        intersection_points = intersections_by_geom.get(idx, [])
        
        logger.debug(f"Processing geometry {idx}: {len(intersection_points)} intersection points")
        for i, pt in enumerate(intersection_points):
            logger.debug(f"  Intersection {i}: {pt}")
        
        if not intersection_points:
            # No intersections, keep original geometry
            split_rows.append(dict(row))
        else:
            # Split geometry at intersection points
            split_geoms = _split_linestring_at_points(geom, intersection_points, config.snap_precision)
            logger.debug(f"Split geometry {idx} into {len(split_geoms)} segments")
            
            # Create new rows for each split segment
            for i, split_geom in enumerate(split_geoms):
                logger.debug(f"  Segment {i}: {split_geom}")
                new_row = dict(row)
                new_row['geometry'] = split_geom
                split_rows.append(new_row)
            
            # Count splits (original geometry becomes multiple)
            splits_created = len(split_geoms) - 1
            split_count += splits_created
            logger.debug(f"Created {splits_created} new splits for geometry {idx}")
    
    stats.split_count = split_count
    logger.info(f"Created {split_count} splits at intersections")
    
    return gpd.GeoDataFrame(split_rows, crs=gdf.crs)


def _find_self_intersections(geom: LineString, tolerance: float) -> List[Point]:
    """Find self-intersection points in a LineString."""
    from shapely.geometry import Point
    
    intersections = []
    coords = list(geom.coords)
    
    # Check each segment against all non-adjacent segments
    for i in range(len(coords) - 1):
        seg1_start = Point(coords[i])
        seg1_end = Point(coords[i + 1])
        seg1 = LineString([coords[i], coords[i + 1]])
        
        for j in range(i + 2, len(coords) - 1):  # Skip adjacent segments
            seg2 = LineString([coords[j], coords[j + 1]])
            
            # Check if segments intersect
            if seg1.intersects(seg2):
                intersection = seg1.intersection(seg2)
                if hasattr(intersection, 'geom_type'):
                    if intersection.geom_type == 'Point':
                        intersections.append(intersection)
    
    return intersections


def _find_intersection_points(geom1: LineString, geom2: LineString, tolerance: float) -> List[Point]:
    """Find intersection points between two LineStrings with tolerance handling."""
    from shapely.geometry import Point, MultiPoint
    
    intersections = []
    
    # First try exact intersection
    if geom1.intersects(geom2):
        intersection = geom1.intersection(geom2)
        
        # Handle different intersection types
        if hasattr(intersection, 'geom_type'):
            if intersection.geom_type == 'Point':
                intersections.append(intersection)
            elif intersection.geom_type == 'MultiPoint':
                intersections.extend(list(intersection.geoms))
            elif intersection.geom_type == 'LineString':
                # Collinear overlap - extract endpoints for splitting
                coords = list(intersection.coords)
                if len(coords) >= 2:
                    intersections.append(Point(coords[0]))
                    intersections.append(Point(coords[-1]))
    
    # If no exact intersections found, try tolerance-based detection
    if not intersections and tolerance > 0:
        intersections = _find_near_intersections(geom1, geom2, tolerance)
    
    return intersections


def _find_near_intersections(geom1: LineString, geom2: LineString, tolerance: float) -> List[Point]:
    """Find near-intersections within tolerance using buffering."""
    from shapely.geometry import Point
    
    intersections = []
    
    # Buffer first geometry by tolerance and check intersection
    buffer1 = geom1.buffer(tolerance)
    if buffer1.intersects(geom2):
        intersection = buffer1.intersection(geom2)
        
        if hasattr(intersection, 'geom_type'):
            if intersection.geom_type in ['Point', 'MultiPoint', 'LineString', 'MultiLineString']:
                # For near-intersections, we need to find the closest points
                # and snap them to grid for consistency
                closest_points = _find_closest_approach_points(geom1, geom2)
                intersections.extend(closest_points)
    
    return intersections


def _find_closest_approach_points(geom1: LineString, geom2: LineString) -> List[Point]:
    """Find the points where two LineStrings come closest to each other."""
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    
    # Find nearest points between the two geometries
    nearest = nearest_points(geom1, geom2)
    if len(nearest) == 2:
        # Use the midpoint between the two nearest points
        p1, p2 = nearest
        midpoint = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        return [midpoint]
    
    return []


def _split_linestring_at_points(geom: LineString, points: List[Point], snap_precision: float) -> List[LineString]:
    """Split a LineString at the given intersection points."""
    from shapely.ops import split
    from shapely.geometry import MultiPoint, LineString
    
    if not points:
        return [geom]
    
    # Use original intersection points without snapping for splitting
    # Snapping can move points away from the geometry, making splitting impossible
    # Remove duplicate points based on original coordinates
    unique_points = []
    seen_coords = set()
    for point in points:
        # Use a small tolerance for coordinate comparison to handle floating point precision
        coord_key = (round(point.x, 6), round(point.y, 6))
        if coord_key not in seen_coords:
            seen_coords.add(coord_key)
            unique_points.append(point)
    
    if not unique_points:
        return [geom]
    
    try:
        # Use shapely split operation
        if len(unique_points) == 1:
            split_result = split(geom, unique_points[0])
        else:
            # Create MultiPoint for multiple split points
            multi_point = MultiPoint(unique_points)
            split_result = split(geom, multi_point)
        
        # Extract LineString geometries from split result
        if hasattr(split_result, 'geoms'):
            # Result is a GeometryCollection
            split_geoms = []
            for g in split_result.geoms:
                if g.geom_type == 'LineString' and not g.is_empty:
                    split_geoms.append(g)
            return split_geoms if split_geoms else [geom]
        else:
            # Split didn't work, return original
            return [geom]
            
    except Exception:
        # If splitting fails, return original geometry
        return [geom]


def find_overlapping_pairs(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig
) -> Iterator[Tuple[int, int]]:
    """Find pairs of potentially overlapping LineStrings using STRtree."""
    logger.info("Finding overlapping pairs with STRtree")
    
    # Create STRtree with buffered geometries
    buffer_dist = config.overlap_tolerance
    buffered_geoms = gdf.geometry.buffer(buffer_dist)
    tree = STRtree(buffered_geoms)
    
    pairs_found = 0
    batch_size = config.pairing_batch_size
    
    for idx in range(len(gdf)):
        # Query potential overlaps
        candidates = tree.query(buffered_geoms.iloc[idx])
        
        for candidate_idx in candidates:
            if candidate_idx <= idx:  # Avoid duplicates and self-pairs
                continue
            
            # Apply pre-filters
            if _passes_pairing_filters(gdf.iloc[idx], gdf.iloc[candidate_idx], config):
                pairs_found += 1
                yield idx, candidate_idx
                
                # Adaptive batching (simplified)
                if pairs_found % batch_size == 0:
                    logger.debug(f"Processed {pairs_found} pairs")


def _passes_pairing_filters(row1: gpd.GeoSeries, row2: gpd.GeoSeries, config: CleanConfig) -> bool:
    """Check if a pair passes the pre-filtering criteria."""
    geom1, geom2 = row1.geometry, row2.geometry
    
    # Length ratio filter
    len1, len2 = geom1.length, geom2.length
    if len1 == 0 or len2 == 0:
        return False
    
    length_ratio = min(len1, len2) / max(len1, len2)
    if not (config.pairing_length_ratio_min <= length_ratio <= config.pairing_length_ratio_max):
        return False
    
    # Additional filters would go here:
    # - Bearing difference
    # - Projected overlap ratio
    # For now, we'll keep it simple
    
    return True


def _group_by_geometry_similarity(geometries, feature_indices, tolerance):
    """Group geometries that are truly similar (not just same endpoints)."""
    groups = []
    used = set()
    
    for i, (geom1, idx1) in enumerate(zip(geometries, feature_indices)):
        if idx1 in used:
            continue
            
        group = [idx1]
        used.add(idx1)
        
        for j, (geom2, idx2) in enumerate(zip(geometries, feature_indices)):
            if i >= j or idx2 in used:
                continue
                
            # Check if geometries are actually similar along their path
            if _geometries_are_similar(geom1, geom2, tolerance):
                group.append(idx2)
                used.add(idx2)
        
        groups.append(group)
    
    return groups


def _geometries_are_similar(geom1, geom2, tolerance):
    """Check if two geometries follow similar paths (not just same endpoints)."""
    from shapely.geometry import LineString
    
    # Quick checks first
    if abs(geom1.length - geom2.length) > tolerance * 2:
        return False
    
    # Sample points along both geometries
    num_samples = min(10, max(3, int(min(geom1.length, geom2.length) / tolerance)))
    
    for i in range(num_samples):
        t = i / (num_samples - 1)
        
        point1 = geom1.interpolate(t, normalized=True)
        point2 = geom2.interpolate(t, normalized=True)
        
        if point1.distance(point2) > tolerance:
            return False
    
    return True


def _would_disconnect_network(gdf, current_keep_indices, idx_to_remove):
    """Test if removing a feature would disconnect the network."""
    # Create temporary set without the feature to be removed
    test_indices = current_keep_indices - {idx_to_remove}
    
    if len(test_indices) <= 1:
        return False  # Can't disconnect a network with ≤1 edges
    
    # Build adjacency list from remaining features
    from collections import defaultdict, deque
    
    adj = defaultdict(set)
    for idx in test_indices:
        if idx >= len(gdf):
            continue
        
        coords = list(gdf.iloc[idx].geometry.coords)
        start = tuple(round(x, 8) for x in coords[0])
        end = tuple(round(x, 8) for x in coords[-1])
        
        adj[start].add(end)
        adj[end].add(start)
    
    if not adj:
        return False
    
    # Test connectivity with BFS
    start_node = next(iter(adj.keys()))
    visited = set()
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    # Check if all nodes are reachable
    all_nodes = set(adj.keys())
    return len(visited) < len(all_nodes)


def deoverlap_linestrings(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig, 
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Remove overlaps and deduplicate LineStrings."""
    logger.info("De-overlapping and deduplicating LineStrings")
    
    overlap_pairs = 0
    duplicates_removed = 0
    
    # Find overlapping pairs
    overlapping_pairs = list(find_overlapping_pairs(gdf, config))
    overlap_pairs = len(overlapping_pairs)
    
    # For now, implement simple deduplication based on geometry equality
    # A full implementation would handle partial overlaps by splitting
    
    # Use QCK graph for deduplication
    from .graph import QCKGraph
    graph = QCKGraph(config, gdf.crs.to_string())
    
    # Add all edges to graph
    feature_to_edge = {}
    for idx, row in gdf.iterrows():
        edge_key = graph.add_edge(row.geometry, idx, dict(row.drop('geometry')))
        feature_to_edge[idx] = edge_key
    
    # Find duplicates and apply deduplication policy
    duplicates = graph.find_duplicates()
    keep_indices = set(range(len(gdf)))
    
    for edge_key, feature_indices in duplicates.items():
        if len(feature_indices) <= 1:
            continue
            
        # Optional: Check geometry path similarity if enabled
        if getattr(config, 'check_geometry_paths', False):
            # Group by actual geometry similarity (not just endpoints)
            geometry_groups = _group_by_geometry_similarity(
                [gdf.iloc[idx].geometry for idx in feature_indices], 
                feature_indices, 
                config.overlap_tolerance
            )
        else:
            # Default: treat all features with same endpoints as one group
            geometry_groups = [feature_indices]
        
        for geom_group in geometry_groups:
            if len(geom_group) <= 1:
                continue
                
            # Apply deduplication policy
            if config.dedup_policy == "keep-shortest":
                # Keep the feature with shortest geometry
                lengths = [(idx, gdf.iloc[idx].geometry.length) for idx in geom_group]
                keep_idx = min(lengths, key=lambda x: x[1])[0]
            elif config.dedup_policy == "keep-first":
                keep_idx = min(geom_group)
            elif config.dedup_policy == "keep-longest":
                lengths = [(idx, gdf.iloc[idx].geometry.length) for idx in geom_group]
                keep_idx = max(lengths, key=lambda x: x[1])[0]
            
            # Remove duplicates
            for idx in geom_group:
                if idx != keep_idx:
                    keep_indices.discard(idx)
                duplicates_removed += 1
    
    # Create result GeoDataFrame with remaining features
    remaining_rows = [dict(gdf.iloc[idx]) for idx in sorted(keep_indices)]
    result_gdf = gpd.GeoDataFrame(remaining_rows, crs=gdf.crs)
    
    stats.overlap_pairs_found = overlap_pairs  
    stats.duplicates_removed = duplicates_removed
    
    logger.info(f"Found {overlap_pairs} overlapping pairs, removed {duplicates_removed} duplicates")
    return result_gdf


def simplify_topology_preserving(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig,
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Apply topology-preserving simplification using global arc sharing."""
    logger.info(f"Applying topology-preserving simplification (tolerance={config.simplify_tolerance}m)")
    
    # Count vertices before simplification
    vertices_before = sum(len(get_coordinates(geom)) for geom in gdf.geometry)
    stats.vertices_before_simplify = vertices_before
    
    # Simple implementation using shapely's simplify
    # A full TopoJSON-style implementation would:
    # 1. Build shared arcs between anchored nodes
    # 2. Simplify arcs while preserving node positions
    # 3. Reassemble edges from simplified arcs
    
    simplified_rows = []
    for idx, row in gdf.iterrows():
        simplified_geom = row.geometry.simplify(
            tolerance=config.simplify_tolerance,
            preserve_topology=True
        )
        
        attributes = dict(row.drop('geometry'))
        simplified_rows.append({'geometry': simplified_geom, **attributes})
    
    result_gdf = gpd.GeoDataFrame(simplified_rows, crs=gdf.crs)
    
    # Count vertices after simplification
    vertices_after = sum(len(get_coordinates(geom)) for geom in result_gdf.geometry)
    stats.vertices_after_simplify = vertices_after
    
    # Prune tiny segments
    if config.prune_epsilon > 0:
        result_gdf = _prune_tiny_segments(result_gdf, config.prune_epsilon, stats)
    
    logger.info(f"Simplified {vertices_before} → {vertices_after} vertices")
    return result_gdf


def _prune_tiny_segments(
    gdf: gpd.GeoDataFrame,
    epsilon: float,
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Remove tiny segments shorter than epsilon."""
    pruned_count = 0
    pruned_rows = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.length < epsilon:
            pruned_count += 1
            continue  # Skip tiny segments
        
        pruned_rows.append(dict(row))
    
    stats.pruned_segments = pruned_count
    logger.info(f"Pruned {pruned_count} tiny segments (< {epsilon}m)")
    
    return gpd.GeoDataFrame(pruned_rows, crs=gdf.crs)


def preserve_and_merge_attributes(
    rows: List[Dict[str, Any]], 
    preserve_fields: Optional[List[str]]
) -> Dict[str, Any]:
    """Merge attributes according to preservation policy.
    
    For fields in preserve_fields (or all if None), concatenate unique values.
    """
    if not rows:
        return {}
    
    if preserve_fields is None:
        # Preserve all fields except geometry and internal fields
        preserve_fields = [
            col for col in rows[0].keys() 
            if col not in ('geometry', '_original_idx', '_part_idx')
        ]
    
    merged_attrs = {}
    
    for field in preserve_fields:
        values = []
        for row in rows:
            if field in row and row[field] is not None:
                val = row[field]
                if val not in values:  # Keep unique values in input order
                    values.append(val)
        
        # Handle concatenation based on type
        if not values:
            merged_attrs[field] = None
        elif len(values) == 1:
            merged_attrs[field] = values[0]
        else:
            # Concatenate multiple values
            if isinstance(values[0], (int, float)):
                # For numeric fields, use comma separation
                merged_attrs[field] = ",".join(str(v) for v in values)
            else:
                # For string fields, use comma + quotes if needed
                str_values = []
                for v in values:
                    s = str(v)
                    if ',' in s:
                        str_values.append(f'"{s}"')
                    else:
                        str_values.append(s)
                merged_attrs[field] = ",".join(str_values)
    
    return merged_attrs


def merge_degree2_nodes(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig,
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Merge LineStrings connected at degree-2 nodes (endpoints with exactly 2 connections).
    
    Args:
        gdf: Input GeoDataFrame with LineString geometries
        config: Configuration with merge_degree2_nodes settings
        stats: Statistics tracker
        
    Returns:
        GeoDataFrame with merged LineStrings
    """
    if not config.merge_degree2_nodes:
        logger.info("Degree-2 node merging disabled")
        return gdf
    
    logger.info(f"Merging degree-2 nodes (max length ratio: {config.merge_max_length_ratio})")
    
    from collections import defaultdict
    from shapely.geometry import LineString
    from shapely.ops import linemerge
    import numpy as np
    
    # Build connectivity graph
    endpoint_to_features = defaultdict(list)
    feature_endpoints = {}
    
    for idx, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        start = tuple(round(x, 8) for x in coords[0])
        end = tuple(round(x, 8) for x in coords[-1])
        
        endpoint_to_features[start].append(idx)
        endpoint_to_features[end].append(idx)
        feature_endpoints[idx] = (start, end)
    
    # Find degree-2 nodes
    degree2_nodes = {
        point: features for point, features in endpoint_to_features.items()
        if len(features) == 2
    }
    
    logger.info(f"Found {len(degree2_nodes)} degree-2 nodes")
    
    # Track which features have been merged
    merged_features = set()
    merged_groups = []
    merges_performed = 0
    
    # Process each degree-2 node
    for node, connected_features in degree2_nodes.items():
        if len(connected_features) != 2:
            continue
            
        feat1_idx, feat2_idx = connected_features
        
        # Skip if either feature already merged
        if feat1_idx in merged_features or feat2_idx in merged_features:
            continue
        
        feat1 = gdf.iloc[feat1_idx]
        feat2 = gdf.iloc[feat2_idx]
        
        # Check length ratio constraint
        len1, len2 = feat1.geometry.length, feat2.geometry.length
        length_ratio = max(len1, len2) / max(min(len1, len2), 1e-10)
        
        if length_ratio > config.merge_max_length_ratio:
            logger.debug(f"Skipping merge: length ratio {length_ratio:.2f} > {config.merge_max_length_ratio}")
            continue
        
        # Check geometric connectivity and try to merge
        merged_geom = _attempt_geometric_merge(feat1.geometry, feat2.geometry, node)
        
        if merged_geom is None:
            logger.debug(f"Skipping merge: geometric merge failed at node {node}")
            continue
        
        # Merge attributes
        merged_attrs = _merge_attributes(feat1, feat2, config.merge_attribute_policy)
        
        # Store merged group
        merged_groups.append({
            'geometry': merged_geom,
            'attributes': merged_attrs,
            'original_indices': [feat1_idx, feat2_idx]
        })
        
        merged_features.update([feat1_idx, feat2_idx])
        merges_performed += 1
    
    # Build result GeoDataFrame
    result_rows = []
    
    # Add merged features
    for group in merged_groups:
        row_data = group['attributes'].copy()
        row_data['geometry'] = group['geometry']
        result_rows.append(row_data)
    
    # Add unmerged features
    for idx, row in gdf.iterrows():
        if idx not in merged_features:
            result_rows.append(dict(row))
    
    result_gdf = gpd.GeoDataFrame(result_rows, crs=gdf.crs)
    
    # Update statistics
    stats.degree2_nodes_found = len(degree2_nodes)
    stats.degree2_merges_performed = merges_performed
    
    logger.info(f"Merged {merges_performed} degree-2 node pairs: {len(gdf)} → {len(result_gdf)} features")
    
    return result_gdf


def _attempt_geometric_merge(geom1: LineString, geom2: LineString, shared_node: tuple) -> LineString | None:
    """Attempt to merge two LineStrings at a shared endpoint.
    
    Args:
        geom1, geom2: LineString geometries to merge
        shared_node: Coordinates of the shared endpoint
        
    Returns:
        Merged LineString or None if merge is not possible
    """
    from shapely.geometry import LineString
    from shapely.ops import linemerge
    
    coords1 = list(geom1.coords)
    coords2 = list(geom2.coords)
    
    # Find which endpoints are shared
    start1 = tuple(round(x, 8) for x in coords1[0])
    end1 = tuple(round(x, 8) for x in coords1[-1])
    start2 = tuple(round(x, 8) for x in coords2[0])
    end2 = tuple(round(x, 8) for x in coords2[-1])
    
    shared_rounded = tuple(round(x, 8) for x in shared_node)
    
    # Determine connection pattern and create merged coordinates
    merged_coords = None
    
    if end1 == shared_rounded and start2 == shared_rounded:
        # geom1 → shared_node → geom2
        merged_coords = coords1 + coords2[1:]  # Skip duplicate shared point
        
    elif end1 == shared_rounded and end2 == shared_rounded:
        # geom1 → shared_node ← geom2 (reverse geom2)
        merged_coords = coords1 + list(reversed(coords2))[1:]  # Skip duplicate shared point
        
    elif start1 == shared_rounded and start2 == shared_rounded:
        # geom2 ← shared_node ← geom1 (reverse geom1)
        merged_coords = list(reversed(coords1)) + coords2[1:]  # Skip duplicate shared point
        
    elif start1 == shared_rounded and end2 == shared_rounded:
        # geom2 → shared_node → geom1
        merged_coords = coords2 + coords1[1:]  # Skip duplicate shared point
    
    if merged_coords is None or len(merged_coords) < 2:
        return None
    
    try:
        # Create merged geometry
        merged_geom = LineString(merged_coords)
        
        # Validate the result
        if not merged_geom.is_valid or merged_geom.is_empty:
            return None
            
        return merged_geom
        
    except Exception as e:
        logger.debug(f"Geometric merge failed: {e}")
        return None


def _merge_attributes(feat1, feat2, policy: str) -> dict:
    """Merge attributes from two features according to the specified policy.
    
    Args:
        feat1, feat2: Feature rows with attributes
        policy: "concatenate", "keep-longest", or "keep-first"
        
    Returns:
        Dictionary of merged attributes
    """
    # Start with feat1 attributes (excluding geometry)
    merged = {k: v for k, v in feat1.items() if k != 'geometry'}
    
    if policy == "keep-first":
        # Keep feat1 attributes, only add missing ones from feat2
        for key, value in feat2.items():
            if key != 'geometry' and key not in merged:
                merged[key] = value
                
    elif policy == "keep-longest":
        # Use attributes from the longer feature
        longer_feat = feat1 if feat1.geometry.length >= feat2.geometry.length else feat2
        merged = {k: v for k, v in longer_feat.items() if k != 'geometry'}
        
    elif policy == "concatenate":
        # Concatenate string attributes, use first value for others
        for key, value2 in feat2.items():
            if key == 'geometry':
                continue
                
            if key in merged:
                value1 = merged[key]
                
                # Concatenate strings if both are strings and different
                if (isinstance(value1, str) and isinstance(value2, str) and 
                    value1 != value2 and value2.strip()):
                    merged[key] = f"{value1}; {value2}"
                # For non-strings or identical values, keep first
                # (could extend this logic for other types)
                    
            else:
                # New attribute from feat2
                merged[key] = value2
    
    return merged


def count_connected_components(gdf: gpd.GeoDataFrame) -> int:
    """Count connected components in the LineString network using BFS.
    
    Args:
        gdf: GeoDataFrame with LineString geometries
        
    Returns:
        Number of connected components
    """
    if len(gdf) == 0:
        return 0
    
    from collections import defaultdict, deque
    
    # Build adjacency list from endpoints
    adj = defaultdict(set)
    all_points = set()
    
    for idx, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        start = tuple(round(x, 8) for x in coords[0])
        end = tuple(round(x, 8) for x in coords[-1])
        
        adj[start].add(end)
        adj[end].add(start)
        all_points.update([start, end])
    
    # Find connected components using BFS
    visited = set()
    component_count = 0
    
    for point in all_points:
        if point not in visited:
            # Start new component
            component_count += 1
            queue = deque([point])
            
            # BFS to find all points in this component
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    
                    # Add all unvisited neighbors
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
    
    return component_count


def remove_small_components(
    gdf: gpd.GeoDataFrame,
    config: CleanConfig,
    stats: ProcessingStats
) -> gpd.GeoDataFrame:
    """Remove connected components smaller than specified thresholds.
    
    Args:
        gdf: Input GeoDataFrame with LineString geometries
        config: Configuration with small component removal settings
        stats: Statistics tracker
        
    Returns:
        GeoDataFrame with small components removed
    """
    if not config.remove_small_components:
        logger.info("Small component removal disabled")
        return gdf
    
    if len(gdf) == 0:
        return gdf
    
    logger.info(f"Removing components with < {config.min_component_features} features or < {config.min_component_length}m total length")
    logger.debug(f"Input GDF columns: {list(gdf.columns)}")
    
    from collections import defaultdict, deque
    
    # Build adjacency list from endpoints
    adj = defaultdict(set)
    all_points = set()
    
    for idx, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        start = tuple(round(x, 8) for x in coords[0])
        end = tuple(round(x, 8) for x in coords[-1])
        
        adj[start].add(end)
        adj[end].add(start)
        all_points.update([start, end])
    
    # Find connected components and group features by component
    visited = set()
    components = []
    point_to_component = {}
    
    for point in all_points:
        if point not in visited:
            # Start new component
            component_points = set()
            queue = deque([point])
            
            # BFS to find all points in this component
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component_points.add(current)
                    point_to_component[current] = len(components)
                    
                    # Add all unvisited neighbors
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            components.append(component_points)
    
    # Group features by component and calculate component metrics
    component_features = [[] for _ in components]
    
    for idx, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        start = tuple(round(x, 8) for x in coords[0])
        # Find which component this feature belongs to
        component_id = point_to_component[start]
        component_features[component_id].append(idx)
    
    # Evaluate each component against size thresholds
    keep_indices = set()
    small_components_removed = 0
    features_removed = 0
    
    for component_id, feature_indices in enumerate(component_features):
        if not feature_indices:
            continue
            
        # Calculate component metrics
        feature_count = len(feature_indices)
        total_length = sum(gdf.iloc[idx].geometry.length for idx in feature_indices)
        
        # Check if component meets minimum thresholds
        meets_feature_threshold = feature_count >= config.min_component_features
        meets_length_threshold = total_length >= config.min_component_length
        
        if meets_feature_threshold and meets_length_threshold:
            # Keep component
            keep_indices.update(feature_indices)
        else:
            # Remove component (it's too small)
            small_components_removed += 1
            features_removed += len(feature_indices)
            
            logger.debug(f"Removing component {component_id}: {feature_count} features, {total_length:.2f}m total length")
    
    # Count small components found (all that are below threshold)
    small_components_found = 0
    for component_id, feature_indices in enumerate(component_features):
        if feature_indices:
            feature_count = len(feature_indices)
            total_length = sum(gdf.iloc[idx].geometry.length for idx in feature_indices)
            if feature_count < config.min_component_features or total_length < config.min_component_length:
                small_components_found += 1
    
    # Update statistics
    stats.small_components_found = small_components_found
    stats.small_components_removed = small_components_removed  
    stats.small_component_features_removed = features_removed
    
    # Create result GeoDataFrame
    if keep_indices:
        # Keep selected rows directly from the original GDF to preserve structure
        result_gdf = gdf.iloc[sorted(keep_indices)].copy()
    else:
        # All components were removed - create empty GeoDataFrame with same schema
        # Use iloc[:0] to preserve all columns including geometry
        result_gdf = gdf.iloc[:0].copy()
    
    logger.info(f"Removed {small_components_removed} small components ({features_removed} features): {len(gdf)} → {len(result_gdf)} features")
    logger.debug(f"Result GDF columns: {list(result_gdf.columns)}")
    logger.debug(f"Result GDF CRS: {result_gdf.crs}")
    
    return result_gdf