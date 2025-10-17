"""Modular small component removal operation for tnclean."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import geopandas as gpd

from ..types import CleanConfig, ProcessingStats
from ..ops import (
    validate_and_clamp_config,
    setup_crs_transform,
    transform_to_metric,
    transform_to_original,
    remove_small_components,
    count_connected_components,
)

logger = logging.getLogger(__name__)


def run_remove_components(
    input_path: str,
    output_path: str,
    config: CleanConfig
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Remove small connected components from LineString network.
    
    Args:
        input_path: Path to input Shapefile or GeoJSON
        output_path: Path for output GeoJSON
        config: Configuration with component removal settings
        
    Returns:
        Tuple of (processed GeoDataFrame, report dict)
        
    Raises:
        ValueError: On invalid geometries, empty results, or validation failures
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
    
    logger.info(f"Starting component removal: {input_path} → {output_path}")
    
    try:
        # Load input data
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading input from {input_path}")
        gdf = gpd.read_file(input_path)
        logger.info(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
        
        if len(gdf) == 0:
            raise ValueError("Input dataset is empty")
        
        if gdf.crs is None:
            raise ValueError("Input dataset has no CRS defined")
        
        # Setup CRS transformation
        original_crs, target_crs, to_metric, to_original = setup_crs_transform(gdf, config)
        
        # Transform to metric CRS if needed
        if to_metric:
            gdf = transform_to_metric(gdf, to_metric, target_crs)
        
        # Validate and clamp configuration
        config, validation_issues = validate_and_clamp_config(gdf, config)
        if validation_issues and config.verbose >= 1:
            for issue in validation_issues:
                logger.warning(f"Config validation: {issue}")
        
        # Track input statistics
        stats.input_count = len(gdf)
        stats.input_total_length = gdf.geometry.length.sum()
        
        # Remove small components
        gdf = remove_small_components(gdf, config, stats)
        
        # Calculate output statistics in metric CRS before transformation
        stats.output_count = len(gdf)
        stats.output_total_length = gdf.geometry.length.sum()
        stats.output_connected_components = count_connected_components(gdf)
        
        # Transform back to original CRS if needed
        if to_original:
            gdf = transform_to_original(gdf, to_original, original_crs)
        
        # Final validation - ensure we still have valid geometries
        if len(gdf) == 0:
            logger.warning("All components were removed - result is empty")
        else:
            invalid_mask = ~gdf.geometry.is_valid
            if invalid_mask.any():
                raise ValueError(f"Output contains {invalid_mask.sum()} invalid geometries")
        
        # Export results
        _export_results(gdf, output_path)
        stats.processing_time = time.time() - start_time
        
        # Generate report
        report = _generate_report(stats, config, original_crs, target_crs, validation_issues)
        
        logger.info(f"Component removal completed in {stats.processing_time:.2f}s")
        logger.info(f"Processed {stats.input_count} → {stats.output_count} features")
        logger.info(f"Removed {stats.small_components_removed} small components")
        
        return gdf, report
        
    except Exception as e:
        logger.error(f"Component removal failed: {e}")
        raise


def _export_results(gdf: gpd.GeoDataFrame, output_path: str) -> None:
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
    """Generate processing report."""
    return {
        # Input/Output counts
        'input_count': stats.input_count,
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
        
        # Small component removal statistics
        'small_components_found': stats.small_components_found,
        'small_components_removed': stats.small_components_removed,
        'small_component_features_removed': stats.small_component_features_removed,
        
        # Performance metrics
        'processing_time': stats.processing_time,
        'memory_peak_mb': stats.memory_peak_mb,
        
        # Configuration and CRS info
        'original_crs': original_crs,
        'processing_crs': target_crs,
        'config': {
            'remove_small_components': config.remove_small_components,
            'min_component_features': config.min_component_features,
            'min_component_length': config.min_component_length,
        },
        
        # Validation issues
        'validation_issues': validation_issues,
        
        # Pipeline metadata  
        'pipeline_version': '0.1.0',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }