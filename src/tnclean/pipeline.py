"""Main pipeline orchestration for tnclean."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import geopandas as gpd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

from .types import CleanConfig, ProcessingStats
from .ops import (
    explode_multilinestrings,
    snap_coordinates,
    validate_and_clamp_config,
    setup_crs_transform,
    transform_to_metric,
    transform_to_original,
    split_at_intersections,
    deoverlap_linestrings,
    merge_degree2_nodes,
    remove_small_components,
    simplify_topology_preserving,
    count_connected_components,
)

logger = logging.getLogger(__name__)
console = Console()


def clean_network(
    input_path: str,
    output_path: str,
    config: CleanConfig = CleanConfig()
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Run the complete tnclean pipeline.
    
    Fixed pipeline order: explode → snap → split → de-overlap → merge degree-2 → remove small components → simplify → export
    
    Args:
        input_path: Path to input Shapefile or GeoJSON
        output_path: Path for output GeoJSON
        config: Configuration parameters
        
    Returns:
        Tuple of (processed GeoDataFrame, report dict)
        
    Raises:
        ValueError: On invalid geometries, empty results, or strict validation failures
        FileNotFoundError: If input file doesn't exist
    """
    start_time = time.time()
    stats = ProcessingStats()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if config.verbose >= 2 else 
              logging.INFO if config.verbose >= 1 else 
              logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting tnclean pipeline: {input_path} → {output_path}")
    
    try:
        # Load input data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not config.progress_bar
        ) as progress:
            
            # Task 1: Load and validate input
            task1 = progress.add_task("Loading input data...", total=None)
            gdf = _load_input_data(input_path)
            _validate_input_data(gdf, config)
            stats.input_count = len(gdf)
            # Note: input_total_length will be calculated after CRS transformation
            progress.update(task1, total=1, completed=1, description="[green]OK[/green] Input data loaded")
            
            # Task 2: Setup CRS transformation
            task2 = progress.add_task("Setting up CRS transformation...", total=None)
            original_crs, target_crs, to_metric, to_original = setup_crs_transform(gdf, config)
            
            # Transform to metric CRS if needed
            if to_metric:
                gdf = transform_to_metric(gdf, to_metric, target_crs)
            
            # Calculate input length in metric CRS for accurate measurements
            stats.input_total_length = gdf.geometry.length.sum()
            
            # Validate and clamp configuration
            config, validation_issues = validate_and_clamp_config(gdf, config)
            if validation_issues and config.verbose >= 1:
                for issue in validation_issues:
                    logger.warning(f"Config validation: {issue}")
            
            progress.update(task2, total=1, completed=1, description="[green]OK[/green] CRS setup complete")
            
            # Task 3: Explode MultiLineStrings
            task3 = progress.add_task("Exploding MultiLineStrings...", total=None)
            gdf = explode_multilinestrings(gdf, stats)
            progress.update(task3, total=1, completed=1, description="[green]OK[/green] MultiLineStrings exploded")
            
            # Task 4: Snap coordinates
            task4 = progress.add_task("Snapping coordinates to grid...", total=None) 
            gdf = snap_coordinates(gdf, config, stats)
            progress.update(task4, total=1, completed=1, description="[green]OK[/green] Coordinates snapped")
            
            # Task 5: Split at intersections (with detailed progress)
            feature_count = len(gdf)
            task5 = progress.add_task(f"Splitting at intersections ({feature_count} features)...", total=None)
            
            # Note: The split_at_intersections function now provides detailed logging
            # including progress updates every 10% for large datasets
            start_time = time.time()
            gdf = split_at_intersections(gdf, config, stats)
            duration = time.time() - start_time
            
            # Mark task complete with summary
            progress.update(task5, total=1, completed=1, 
                          description=f"[green]OK[/green] Split at intersections ({stats.split_count} splits in {duration:.1f}s)")
            
            # Task 6: De-overlap and deduplicate
            task6 = progress.add_task("De-overlapping and deduplicating...", total=None)
            gdf = deoverlap_linestrings(gdf, config, stats)
            progress.update(task6, total=1, completed=1, description="[green]OK[/green] Overlaps removed")
            
            # Task 7: Merge degree-2 nodes
            task7 = progress.add_task("Merging degree-2 nodes...", total=None)
            gdf = merge_degree2_nodes(gdf, config, stats)
            if config.merge_degree2_nodes and stats.degree2_merges_performed > 0:
                progress.update(task7, total=1, completed=1, 
                              description=f"[green]OK[/green] Merged {stats.degree2_merges_performed} degree-2 nodes")
            else:
                progress.update(task7, total=1, completed=1, description="[green]OK[/green] Degree-2 merging skipped")
            
            # Task 8: Remove small components
            task8 = progress.add_task("Removing small components...", total=None)
            gdf = remove_small_components(gdf, config, stats)
            if config.remove_small_components and stats.small_components_removed > 0:
                progress.update(task8, total=1, completed=1,
                              description=f"[green]OK[/green] Removed {stats.small_components_removed} small components")
            else:
                progress.update(task8, total=1, completed=1, description="[green]OK[/green] Small component removal skipped")
            
            # Task 9: Topology-preserving simplification
            task9 = progress.add_task("Simplifying geometries...", total=None)
            gdf = simplify_topology_preserving(gdf, config, stats)
            progress.update(task9, total=1, completed=1, description="[green]OK[/green] Geometries simplified")
            
            # Task 10: Transform back and export
            task10 = progress.add_task("Exporting results...", total=None)
            
            # Calculate output statistics in metric CRS before transformation
            stats.output_count = len(gdf)
            stats.output_total_length = gdf.geometry.length.sum()
            stats.output_connected_components = count_connected_components(gdf)
            
            # Transform back to original CRS
            if to_original:
                gdf = transform_to_original(gdf, to_original, original_crs)
            
            # Final validation
            _validate_output_data(gdf, config)
            
            # Export to GeoJSON
            _export_results(gdf, output_path, config)
            
            progress.update(task10, total=1, completed=1, description="[green]OK[/green] Results exported")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    # Calculate final statistics
    stats.processing_time = time.time() - start_time
    
    # Generate report
    report = _generate_report(stats, config, original_crs, target_crs, validation_issues)
    
    logger.info(f"Pipeline completed in {stats.processing_time:.2f}s")
    logger.info(f"Processed {stats.input_count} → {stats.output_count} features")
    logger.info(f"Network topology: {stats.output_connected_components} connected component(s)")
    
    return gdf, report


def _load_input_data(input_path: str) -> gpd.GeoDataFrame:
    """Load input data from Shapefile or GeoJSON."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading input from {input_path}")
    
    try:
        gdf = gpd.read_file(input_path)
        logger.info(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
        return gdf
    except Exception as e:
        raise ValueError(f"Failed to load input file: {e}")


def _validate_input_data(gdf: gpd.GeoDataFrame, config: CleanConfig) -> None:
    """Validate input data meets requirements."""
    if len(gdf) == 0:
        raise ValueError("Input dataset is empty")
    
    if gdf.crs is None:
        raise ValueError("Input dataset has no CRS defined")
    
    # Check geometry types
    valid_types = {'LineString', 'MultiLineString'}
    actual_types = set(gdf.geometry.geom_type.unique())
    invalid_types = actual_types - valid_types
    
    if invalid_types:
        if config.strict_validation:
            raise ValueError(f"Invalid geometry types found: {invalid_types}. Only LineString and MultiLineString are supported.")
        else:
            logger.warning(f"Skipping unsupported geometry types: {invalid_types}")
    
    # Check for invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        if config.strict_validation:
            raise ValueError(f"Found {invalid_count} invalid geometries")
        else:
            logger.warning(f"Found {invalid_count} invalid geometries - may cause issues")


def _validate_output_data(gdf: gpd.GeoDataFrame, config: CleanConfig) -> None:
    """Validate output data meets requirements."""
    if len(gdf) == 0:
        raise ValueError("Pipeline produced empty result - this should not happen")
    
    # Ensure all geometries are LineStrings
    non_linestring = gdf[gdf.geometry.geom_type != 'LineString']
    if len(non_linestring) > 0:
        raise ValueError(f"Output contains {len(non_linestring)} non-LineString geometries")
    
    # Check for invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        raise ValueError(f"Output contains {invalid_mask.sum()} invalid geometries")


def _export_results(gdf: gpd.GeoDataFrame, output_path: str, config: CleanConfig) -> None:
    """Export results to GeoJSON format."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {len(gdf)} features to {output_path}")
    
    # Ensure RFC 7946 compliance
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        # GeoJSON should be in WGS84 for RFC 7946 compliance
        export_gdf = gdf.to_crs('EPSG:4326')
    else:
        export_gdf = gdf
    
    # Remove internal fields
    columns_to_drop = [col for col in export_gdf.columns if col.startswith('_')]
    if columns_to_drop:
        export_gdf = export_gdf.drop(columns=columns_to_drop)
    
    try:
        export_gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Successfully exported to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to export results: {e}")


def _generate_report(
    stats: ProcessingStats,
    config: CleanConfig,
    original_crs: str,
    target_crs: str,
    validation_issues: list
) -> Dict[str, Any]:
    """Generate comprehensive processing report."""
    return {
        # Input/Output counts
        'input_count': stats.input_count,
        'input_multilinestring_count': stats.input_multilinestring_count,
        'output_count': stats.output_count,
        'output_connected_components': stats.output_connected_components,
        
        # Lengths
        'input_total_length': stats.input_total_length,
        'output_total_length': stats.output_total_length,
        'length_change': stats.output_total_length - stats.input_total_length,
        'length_change_percent': (
            (stats.output_total_length - stats.input_total_length) / stats.input_total_length * 100
            if stats.input_total_length > 0 else 0
        ),
        
        # Processing statistics
        'exploded_count': stats.exploded_count,
        'snapped_vertex_changes': stats.snapped_vertex_changes,
        'split_count': stats.split_count,
        'overlap_pairs_found': stats.overlap_pairs_found,
        'duplicates_removed': stats.duplicates_removed,
        
        # Degree-2 node removal statistics
        'degree2_nodes_found': stats.degree2_nodes_found,
        'degree2_merges_performed': stats.degree2_merges_performed,
        
        # Small component removal statistics
        'small_components_found': stats.small_components_found,
        'small_components_removed': stats.small_components_removed,
        'small_component_features_removed': stats.small_component_features_removed,
        
        # Simplification statistics
        'vertices_before_simplify': stats.vertices_before_simplify,
        'vertices_after_simplify': stats.vertices_after_simplify,
        'vertices_removed': stats.vertices_before_simplify - stats.vertices_after_simplify,
        'vertex_reduction_percent': (
            (stats.vertices_before_simplify - stats.vertices_after_simplify) / stats.vertices_before_simplify * 100
            if stats.vertices_before_simplify > 0 else 0
        ),
        'pruned_segments': stats.pruned_segments,
        
        # Performance metrics
        'processing_time': stats.processing_time,
        'memory_peak_mb': stats.memory_peak_mb,
        
        # Configuration and CRS info
        'original_crs': original_crs,
        'processing_crs': target_crs,
        'config': {
            'snap_precision': config.snap_precision,
            'overlap_tolerance': config.overlap_tolerance,
            'simplify_tolerance': config.simplify_tolerance,
            'prune_epsilon': config.prune_epsilon,
            'dedup_policy': config.dedup_policy,
            'keep_parallel_edges': config.keep_parallel_edges,
        },
        
        # Validation issues
        'validation_issues': validation_issues,
        
        # Pipeline metadata  
        'pipeline_version': '0.1.0',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }