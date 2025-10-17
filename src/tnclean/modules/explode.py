"""Explode MultiLineString geometries into separate LineStrings."""

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import geopandas as gpd

from ..types import CleanConfig, ProcessingStats
from ..ops import explode_multilinestrings

logger = logging.getLogger(__name__)


def run_explode(
    input_path: str,
    output_path: str,
    config: CleanConfig = CleanConfig()
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Explode MultiLineString geometries into separate LineStrings.
    
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
    
    logger.info(f"Starting explode operation: {input_path} → {output_path}")
    
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
        
        # Track input stats (length in original CRS - informational only)
        stats.input_count = len(gdf)
        # Note: Length calculated in original CRS - this is informational only for explode operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            stats.input_total_length = gdf.geometry.length.sum()
        
        # Check geometry types
        valid_types = {'LineString', 'MultiLineString'}
        actual_types = set(gdf.geometry.geom_type.unique())
        invalid_types = actual_types - valid_types
        
        if invalid_types:
            if config.strict_validation:
                raise ValueError(f"Invalid geometry types found: {invalid_types}")
            else:
                logger.warning(f"Skipping unsupported geometry types: {invalid_types}")
        
        # Explode MultiLineStrings
        logger.info("Exploding MultiLineString geometries...")
        gdf = explode_multilinestrings(gdf, stats)
        
        # Track output stats (length in original CRS - informational only)
        stats.output_count = len(gdf)
        # Note: Length calculated in original CRS - this is informational only for explode operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            stats.output_total_length = gdf.geometry.length.sum()
        
        # Export results
        _export_results(gdf, output_path, config)
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        # Generate report
        report = {
            'input_count': stats.input_count,
            'input_multilinestring_count': stats.input_multilinestring_count,
            'exploded_count': stats.exploded_count,
            'output_count': stats.output_count,
            'input_total_length': stats.input_total_length,
            'output_total_length': stats.output_total_length,
            'processing_time': stats.processing_time,
            'operation': 'explode',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        }
        
        logger.info(f"Explode completed in {stats.processing_time:.2f}s")
        logger.info(f"Processed {stats.input_count} → {stats.output_count} features")
        logger.info(f"Exploded {stats.input_multilinestring_count} MultiLineStrings into {stats.exploded_count} LineStrings")
        
        return gdf, report
        
    except Exception as e:
        logger.error(f"Explode operation failed: {e}")
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