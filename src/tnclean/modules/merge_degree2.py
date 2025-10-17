"""Modular command for degree-2 node removal."""

import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import geopandas as gpd

from ..types import CleanConfig, ProcessingStats
from ..ops import (
    setup_crs_transform,
    transform_to_metric,
    transform_to_original,
    validate_and_clamp_config,
    merge_degree2_nodes,
)

logger = logging.getLogger(__name__)


def run_merge_degree2(
    input_path: str,
    output_path: str,
    config: CleanConfig = CleanConfig()
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Run degree-2 node merging as a standalone operation.
    
    Args:
        input_path: Path to input GeoJSON/Shapefile (should contain LineStrings)
        output_path: Path for output GeoJSON
        config: Configuration parameters
        
    Returns:
        Tuple of (processed GeoDataFrame, report dict)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input is invalid or processing fails
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
    
    logger.info(f"Starting degree-2 node merging: {input_path} → {output_path}")
    
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
        valid_types = {'LineString'}  # Degree-2 merge requires LineStrings only
        actual_types = set(gdf.geometry.geom_type.unique())
        invalid_types = actual_types - valid_types
        
        if invalid_types:
            if config.strict_validation:
                raise ValueError(f"Invalid geometry types found: {invalid_types}. Degree-2 merge requires LineString geometries only.")
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
        
        # Validate and clamp configuration to handle "auto" values
        config, validation_issues = validate_and_clamp_config(gdf, config)
        
        # Merge degree-2 nodes
        logger.info(f"Merging degree-2 nodes with max length ratio {config.merge_max_length_ratio} and {config.merge_attribute_policy} policy...")
        gdf = merge_degree2_nodes(gdf, config, stats)
        
        # Track output stats in metric CRS
        stats.output_count = len(gdf)
        stats.output_total_length = gdf.geometry.length.sum()
        
        # Transform back to original CRS
        if to_original:
            logger.info("Transforming back to original CRS")
            gdf = transform_to_original(gdf, to_original, original_crs)
        
        # Export results
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(gdf)} features to {output_path}")
        
        # Ensure RFC 7946 compliance
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            export_gdf = gdf.to_crs('EPSG:4326')
        else:
            export_gdf = gdf
        
        export_gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Successfully exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Degree-2 merge operation failed: {e}")
        raise
    
    # Calculate final statistics
    stats.processing_time = time.time() - start_time
    
    # Generate report
    report = {
        # Input/Output
        'input_count': stats.input_count,
        'output_count': stats.output_count,
        'input_total_length': stats.input_total_length,
        'output_total_length': stats.output_total_length,
        
        # Degree-2 specific statistics
        'degree2_nodes_found': stats.degree2_nodes_found,
        'degree2_merges_performed': stats.degree2_merges_performed,
        
        # Performance
        'processing_time': stats.processing_time,
        
        # Configuration
        'original_crs': original_crs,
        'processing_crs': target_crs,
        'merge_degree2_nodes': config.merge_degree2_nodes,
        'merge_attribute_policy': config.merge_attribute_policy,
        'merge_max_length_ratio': config.merge_max_length_ratio,
        
        # Metadata
        'operation': 'merge_degree2',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }
    
    logger.info(f"Degree-2 merging completed in {stats.processing_time:.2f}s")
    logger.info(f"Processed {stats.input_count} → {stats.output_count} features")
    logger.info(f"Found {stats.degree2_nodes_found} degree-2 nodes")
    logger.info(f"Performed {stats.degree2_merges_performed} merges")
    
    return gdf, report