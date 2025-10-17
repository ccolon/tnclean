"""Split edges at intersections."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import geopandas as gpd

from ..types import CleanConfig, ProcessingStats
from ..ops import (
    split_at_intersections,
    setup_crs_transform,
    transform_to_metric,
    transform_to_original,
    validate_and_clamp_config
)

logger = logging.getLogger(__name__)


def run_split(
    input_path: str,
    output_path: str,
    config: CleanConfig = CleanConfig()
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Split edges at intersections.
    
    Args:
        input_path: Path to input Shapefile or GeoJSON
        output_path: Path for output GeoJSON
        config: Configuration parameters
        
    Returns:
        Tuple of (processed GeoDataFrame, report dict)
        
    Raises:
        ValueError: On invalid geometries or empty results
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
    
    logger.info(f"Starting split operation: {input_path} → {output_path}")
    
    try:
        # Load input data
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        gdf = gpd.read_file(input_path)
        logger.info(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
        
        # Validate input data
        if len(gdf) == 0:
            raise ValueError("Input dataset is empty")
        
        if gdf.crs is None:
            raise ValueError("Input dataset has no CRS defined")
        
        # Check geometry types
        valid_types = {'LineString'}  # Split operation requires LineStrings only
        actual_types = set(gdf.geometry.geom_type.unique())
        invalid_types = actual_types - valid_types
        
        if invalid_types:
            if config.strict_validation:
                raise ValueError(f"Invalid geometry types found: {invalid_types}. Split operation requires LineString geometries only.")
            else:
                logger.warning(f"Skipping unsupported geometry types: {invalid_types}")
                # Filter to valid geometries
                gdf = gdf[gdf.geometry.geom_type.isin(valid_types)]
        
        # Setup CRS transformation
        logger.info("Setting up CRS transformation...")
        original_crs, target_crs, to_metric, to_original = setup_crs_transform(gdf, config)
        
        # Transform to metric CRS if needed
        if to_metric:
            gdf = transform_to_metric(gdf, to_metric, target_crs)
        
        # Track input stats in metric CRS
        stats.input_count = len(gdf)
        stats.input_total_length = gdf.geometry.length.sum()
        
        # Validate and clamp configuration
        config, validation_issues = validate_and_clamp_config(gdf, config)
        if validation_issues and config.verbose >= 1:
            for issue in validation_issues:
                logger.warning(f"Config validation: {issue}")
        
        # Split at intersections
        logger.info("Splitting edges at intersections...")
        gdf = split_at_intersections(gdf, config, stats)
        
        # Track output stats in metric CRS
        stats.output_count = len(gdf)
        stats.output_total_length = gdf.geometry.length.sum()
        
        # Transform back to original CRS
        if to_original:
            gdf = transform_to_original(gdf, to_original, original_crs)
        
        # Export results
        _export_results(gdf, output_path, config)
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        # Generate report
        report = {
            'input_count': stats.input_count,
            'output_count': stats.output_count,
            'input_total_length': stats.input_total_length,
            'output_total_length': stats.output_total_length,
            'split_count': stats.split_count,
            'processing_time': stats.processing_time,
            'original_crs': original_crs,
            'processing_crs': target_crs,
            'validation_issues': validation_issues,
            'operation': 'split',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        }
        
        logger.info(f"Split completed in {stats.processing_time:.2f}s")
        logger.info(f"Processed {stats.input_count} → {stats.output_count} features")
        logger.info(f"Created {stats.split_count} intersection splits")
        
        return gdf, report
        
    except Exception as e:
        logger.error(f"Split operation failed: {e}")
        raise


def _export_results(gdf: gpd.GeoDataFrame, output_path: str, config: CleanConfig) -> None:
    """Export results to GeoJSON format."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {len(gdf)} features to {output_path}")
    
    # Ensure RFC 7946 compliance
    if gdf.crs and gdf.crs.to_epsg() != 4326:
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