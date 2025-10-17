"""Command-line interface for tnclean."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .pipeline import clean_network
from .types import CleanConfig
from .modules import run_explode, run_snap, run_split, run_deoverlap, run_simplify, run_remove_components

app = typer.Typer(
    name="tnclean",
    help="Transport Network Cleaner - clean & simplify transport network lines",
    no_args_is_help=True,
)
console = Console()


@app.command()
def clean(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    
    # CRS options
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing ('auto' to select automatically)"),
    
    # Tolerance parameters (in meters when using metric CRS)
    snap_precision: str = typer.Option("auto", "--snap-precision", help="Grid size for coordinate quantization (meters) or 'auto'"),
    overlap_tolerance: float = typer.Option(0.5, "--overlap-tolerance", help="Proximity tolerance for overlap detection (meters)"),
    simplify_tolerance: float = typer.Option(2.0, "--simplify-tolerance", help="Geometric tolerance for simplification (meters)"),
    prune_epsilon: float = typer.Option(0.1, "--prune-epsilon", help="Threshold for pruning tiny segments (meters)"),
    
    # Processing options
    dedup_policy: str = typer.Option("keep-shortest", "--dedup-policy", help="How to resolve duplicates: keep-shortest, keep-first, keep-longest"),
    keep_parallel_edges: bool = typer.Option(False, "--keep-parallel/--no-keep-parallel", help="Keep parallel edges even when endpoints match"),
    preserve_fields: Optional[List[str]] = typer.Option(None, "--preserve-field", help="Field names to preserve (repeat for multiple fields)"),
    
    # Degree-2 node removal
    merge_degree2_nodes: bool = typer.Option(True, "--merge-degree2/--no-merge-degree2", help="Merge LineStrings connected at degree-2 nodes"),
    merge_attribute_policy: str = typer.Option("concatenate", "--merge-policy", help="Attribute merge policy: concatenate, keep-longest, keep-first"),
    merge_max_length_ratio: float = typer.Option(10.0, "--max-length-ratio", help="Maximum length ratio between merged segments"),
    
    # Small component removal
    remove_small_components: bool = typer.Option(True, "--remove-small-components/--keep-small-components", help="Remove isolated small components"),
    min_component_features: int = typer.Option(1, "--min-component-features", help="Minimum features per component to preserve"),
    min_component_length: float = typer.Option(5.0, "--min-component-length", help="Minimum total length (meters) per component to preserve"),
    
    # Pairing performance options
    pairing_batch_size: int = typer.Option(100000, "--pairing-batch-size", help="Batch size for pairing queries"),
    pairing_bearing_delta_deg: float = typer.Option(5.0, "--pairing-bearing-delta", help="Bearing difference threshold for pairing (degrees)"),
    pairing_length_ratio_min: float = typer.Option(0.5, "--pairing-length-ratio-min", help="Minimum length ratio for pairing"),
    pairing_length_ratio_max: float = typer.Option(2.0, "--pairing-length-ratio-max", help="Maximum length ratio for pairing"),
    pairing_overlap_alpha: float = typer.Option(0.9, "--pairing-overlap-alpha", help="Overlap ratio threshold for pairing"),
    
    # Validation and output options
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on tolerance validation issues"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (use -v, -vv, -vvv)"),
    
    # Configuration file
    config_file: Optional[str] = typer.Option(None, "--config", help="Load configuration from YAML or JSON file"),
    
    # Report output
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Clean and simplify transport network lines.
    
    This command applies a fixed pipeline: explode -> snap -> split -> de-overlap -> merge degree-2 -> remove small components -> simplify -> export
    
    All distance tolerances are in meters when using a metric CRS.
    The 'auto' metric CRS option selects an appropriate projected coordinate system.
    
    Examples:
    
        # Basic usage with auto CRS selection
        tnclean input.geojson output.geojson
        
        # Custom tolerances
        tnclean roads.shp clean_roads.geojson --snap-precision 0.5 --overlap-tolerance 0.25
        
        # Preserve specific fields
        tnclean network.geojson clean.geojson --preserve-field highway --preserve-field name
        
        # Use configuration file
        tnclean input.geojson output.geojson --config config.yaml --report report.json
        
        # Strict validation mode
        tnclean input.geojson output.geojson --strict --verbose
    """
    try:
        # Load configuration from file if provided
        if config_file:
            config = load_config_file(config_file)
            console.print(f"Loaded configuration from {config_file}")
        else:
            config = CleanConfig()
        
        # Override config with command-line arguments
        if metric_crs != "auto":
            config.metric_crs = metric_crs
        # Handle snap_precision: auto or numeric
        if snap_precision == "auto":
            config.snap_precision = "auto"
        else:
            try:
                config.snap_precision = float(snap_precision)
            except ValueError:
                raise typer.BadParameter(f"snap_precision must be 'auto' or a number, got: {snap_precision}")
        config.overlap_tolerance = overlap_tolerance
        config.simplify_tolerance = simplify_tolerance
        config.prune_epsilon = prune_epsilon
        config.dedup_policy = dedup_policy  # type: ignore
        config.keep_parallel_edges = keep_parallel_edges
        config.preserve_fields = preserve_fields
        config.merge_degree2_nodes = merge_degree2_nodes
        config.merge_attribute_policy = merge_attribute_policy  # type: ignore
        config.merge_max_length_ratio = merge_max_length_ratio
        config.remove_small_components = remove_small_components
        config.min_component_features = min_component_features
        config.min_component_length = min_component_length
        config.pairing_batch_size = pairing_batch_size
        config.pairing_bearing_delta_deg = pairing_bearing_delta_deg
        config.pairing_length_ratio_min = pairing_length_ratio_min
        config.pairing_length_ratio_max = pairing_length_ratio_max
        config.pairing_overlap_alpha = pairing_overlap_alpha
        config.strict_validation = strict_validation
        config.progress_bar = progress
        config.verbose = verbose
        
        # Display configuration if verbose
        if verbose >= 1:
            display_config(config)
        
        # Run the pipeline
        console.print("Starting tnclean pipeline...")
        gdf, report = clean_network(input_path, output_path, config)
        
        # Display results
        display_results(report, verbose)
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
        
        console.print("Pipeline completed successfully!")
        
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command()
def validate(
    input_path: str = typer.Argument(..., help="Input file to validate"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Configuration file to validate against"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation info"),
) -> None:
    """Validate input data and configuration without processing."""
    try:
        import geopandas as gpd
        from .ops import validate_and_clamp_config, setup_crs_transform
        
        console.print(f"Validating {input_path}")
        
        # Load data
        gdf = gpd.read_file(input_path)
        console.print(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
        
        # Load config
        if config_file:
            config = load_config_file(config_file)
        else:
            config = CleanConfig()
        
        # Setup CRS transformation to get metric CRS
        original_crs, target_crs, to_metric, to_original = setup_crs_transform(gdf, config)
        if to_metric:
            gdf = gdf.to_crs(target_crs)
        
        # Validate configuration
        validated_config, issues = validate_and_clamp_config(gdf, config)
        
        # Display validation results
        if not issues:
            console.print("Validation passed - no issues found", style="bold green")
        else:
            console.print(f"Found {len(issues)} validation issues:", style="bold yellow")
            for i, issue in enumerate(issues, 1):
                console.print(f"  {i}. {issue}")
        
        if verbose:
            # Show dataset statistics
            from .graph import compute_dataset_stats
            from .types import ValidationBounds
            
            vertex_spacings, edge_lengths, bbox_diagonal = compute_dataset_stats(gdf)
            bounds = ValidationBounds.compute_bounds(vertex_spacings, edge_lengths, bbox_diagonal)
            
            table = Table(title="Dataset Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Features", str(len(gdf)))
            table.add_row("Total length", f"{gdf.geometry.length.sum():.1f}m")
            table.add_row("Bbox diagonal", f"{bbox_diagonal:.1f}m")
            table.add_row("Median edge length", f"{bounds.median_edge_length:.3f}m")
            table.add_row("Vertex spacing P05", f"{bounds.vertex_spacing_p05:.3f}m")
            table.add_row("Vertex spacing P50", f"{bounds.vertex_spacing_p50:.3f}m")
            
            console.print(table)
        
    except Exception as e:
        console.print(f"Validation failed: {e}", style="bold red")
        sys.exit(1)


@app.command()
def config(
    output_path: str = typer.Argument(..., help="Output path for configuration file"),
    format: str = typer.Option("yaml", "--format", help="Output format: yaml or json"),
) -> None:
    """Generate a default configuration file."""
    try:
        config = CleanConfig()
        config_dict = {
            'metric_crs': config.metric_crs,
            'snap_precision': config.snap_precision,
            'overlap_tolerance': config.overlap_tolerance,
            'simplify_tolerance': config.simplify_tolerance,
            'prune_epsilon': config.prune_epsilon,
            'dedup_policy': config.dedup_policy,
            'keep_parallel_edges': config.keep_parallel_edges,
            'preserve_fields': list(config.preserve_fields) if config.preserve_fields else None,
            'pairing_batch_size': config.pairing_batch_size,
            'pairing_bearing_delta_deg': config.pairing_bearing_delta_deg,
            'pairing_length_ratio_min': config.pairing_length_ratio_min,
            'pairing_length_ratio_max': config.pairing_length_ratio_max,
            'pairing_overlap_alpha': config.pairing_overlap_alpha,
            'strict_validation': config.strict_validation,
            'progress_bar': config.progress_bar,
            'verbose': config.verbose,
        }
        
        output_file = Path(output_path)
        
        if format.lower() == 'yaml':
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        console.print(f"Default configuration saved to {output_path}")
        
    except Exception as e:
        console.print(f"Failed to create config file: {e}", style="bold red")
        sys.exit(1)


def load_config_file(config_path: str) -> CleanConfig:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path) as f:
        if path.suffix.lower() in ['.yml', '.yaml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    # Create CleanConfig with loaded values
    return CleanConfig(**config_dict)


def display_config(config: CleanConfig) -> None:
    """Display current configuration in a formatted table."""
    table = Table(title="Configuration", show_header=True, header_style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Metric CRS", str(config.metric_crs))
    table.add_row("Snap precision", f"{config.snap_precision}m")
    table.add_row("Overlap tolerance", f"{config.overlap_tolerance}m")
    table.add_row("Simplify tolerance", f"{config.simplify_tolerance}m")
    table.add_row("Prune epsilon", f"{config.prune_epsilon}m")
    table.add_row("Dedup policy", config.dedup_policy)
    table.add_row("Keep parallel edges", str(config.keep_parallel_edges))
    table.add_row("Preserve fields", str(config.preserve_fields))
    table.add_row("Strict validation", str(config.strict_validation))
    
    console.print(table)


def display_results(report: dict, verbose: int) -> None:
    """Display processing results."""
    # Summary panel
    summary_text = (
        f"Input:  {report['input_count']} features, {report['input_total_length']:.1f}m total length\n"
        f"Output: {report['output_count']} features, {report['output_total_length']:.1f}m total length\n"
        f"Network topology: {report['output_connected_components']} connected component(s)\n"
        f"Processing time: {report['processing_time']:.2f}s"
    )
    
    console.print(Panel(summary_text, title="Processing Summary", expand=False))
    
    if verbose >= 1:
        # Detailed statistics table
        table = Table(title="Detailed Statistics", show_header=True)
        table.add_column("Operation", style="cyan")
        table.add_column("Count", style="magenta")
        
        table.add_row("MultiLineStrings exploded", str(report['input_multilinestring_count']))
        table.add_row("Features exploded to", str(report['exploded_count']))
        table.add_row("Vertices snapped", str(report['snapped_vertex_changes']))
        table.add_row("Intersection splits", str(report['split_count']))
        table.add_row("Overlapping pairs found", str(report['overlap_pairs_found']))
        table.add_row("Duplicates removed", str(report['duplicates_removed']))
        table.add_row("Degree-2 nodes found", str(report['degree2_nodes_found']))
        table.add_row("Degree-2 merges performed", str(report['degree2_merges_performed']))
        table.add_row("Small components found", str(report['small_components_found']))
        table.add_row("Small components removed", str(report['small_components_removed']))
        table.add_row("Vertices before simplify", str(report['vertices_before_simplify']))
        table.add_row("Vertices after simplify", str(report['vertices_after_simplify']))
        table.add_row("Vertices removed", str(report['vertices_removed']))
        table.add_row("Segments pruned", str(report['pruned_segments']))
        
        console.print(table)
    
    if verbose >= 2 and report.get('validation_issues'):
        console.print("Validation issues (auto-corrected):", style="yellow")
        for issue in report['validation_issues']:
            console.print(f"  - {issue}")


def save_report(report: dict, report_path: str) -> None:
    """Save processing report to JSON file."""
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


@app.command()
def explode(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (use -v, -vv, -vvv)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Explode MultiLineString geometries into separate LineStrings."""
    try:
        config = CleanConfig(
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        console.print("Starting explode operation...")
        gdf, report = run_explode(input_path, output_path, config)
        
        # Display results
        console.print(f"Exploded {report['input_multilinestring_count']} MultiLineStrings into {report['exploded_count']} LineStrings")
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features in {report['processing_time']:.2f}s")
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
        
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command()
def snap(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    snap_precision: str = typer.Option("auto", "--snap-precision", help="Grid size for coordinate quantization (meters) or 'auto'"),
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing ('auto' to select automatically)"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Snap coordinates to metric grid."""
    try:
        config = CleanConfig(
            metric_crs=metric_crs,
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        # Handle snap_precision: auto or numeric
        if snap_precision == "auto":
            config.snap_precision = "auto"
        else:
            try:
                config.snap_precision = float(snap_precision)
            except ValueError:
                raise typer.BadParameter(f"snap_precision must be 'auto' or a number, got: {snap_precision}")
        
        console.print("Starting snap operation...")
        gdf, report = run_snap(input_path, output_path, config)
        
        # Display results
        console.print(f"Snapped {report['snapped_vertex_changes']} vertex coordinates to {report['snap_precision']}m grid")
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features in {report['processing_time']:.2f}s")
        
        if report.get('validation_issues') and verbose >= 1:
            console.print("Validation issues (auto-corrected):", style="yellow")
            for issue in report['validation_issues']:
                console.print(f"  - {issue}")
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
        
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command()
def split(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Split edges at intersections."""
    try:
        config = CleanConfig(
            metric_crs=metric_crs,
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        console.print("Starting split operation...")
        gdf, report = run_split(input_path, output_path, config)
        
        # Display results
        console.print(f"Created {report['split_count']} intersection splits")
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features in {report['processing_time']:.2f}s")
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
        
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command()
def deoverlap(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    overlap_tolerance: float = typer.Option(0.5, "--overlap-tolerance", help="Proximity tolerance for overlap detection (meters)"),
    dedup_policy: str = typer.Option("keep-shortest", "--dedup-policy", help="How to resolve duplicates: keep-shortest, keep-first, keep-longest"),
    keep_parallel_edges: bool = typer.Option(False, "--keep-parallel/--no-keep-parallel", help="Keep parallel edges even when endpoints match"),
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """De-overlap and deduplicate linestrings."""
    try:
        config = CleanConfig(
            metric_crs=metric_crs,
            overlap_tolerance=overlap_tolerance,
            dedup_policy=dedup_policy,  # type: ignore
            keep_parallel_edges=keep_parallel_edges,
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        console.print("Starting deoverlap operation...")
        gdf, report = run_deoverlap(input_path, output_path, config)
        
        # Display results
        console.print(f"Found {report['overlap_pairs_found']} overlapping pairs, removed {report['duplicates_removed']} duplicates")
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features in {report['processing_time']:.2f}s")
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
        
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command("merge-degree2")
def merge_degree2_cmd(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    merge_attribute_policy: str = typer.Option("concatenate", "--merge-policy", help="Attribute merge policy: concatenate, keep-longest, keep-first"),
    merge_max_length_ratio: float = typer.Option(10.0, "--max-length-ratio", help="Maximum length ratio between merged segments"),
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Merge LineStrings connected at degree-2 nodes (endpoints with exactly 2 connections)."""
    try:
        from .modules import run_merge_degree2
        
        config = CleanConfig(
            metric_crs=metric_crs,
            merge_degree2_nodes=True,  # Enable degree-2 merging
            merge_attribute_policy=merge_attribute_policy,
            merge_max_length_ratio=merge_max_length_ratio,
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        console.print("Starting degree-2 node merging...")
        
        # Run the operation
        gdf, report = run_merge_degree2(input_path, output_path, config)
        
        # Display results
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features")
        console.print(f"Found {report['degree2_nodes_found']} degree-2 nodes")
        console.print(f"Performed {report['degree2_merges_performed']} merges")
        console.print(f"Processing completed in {report['processing_time']:.2f}s")
        
        # Save report if requested
        if report_path:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            console.print(f"Report saved to {report_path}")
            
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command()
def simplify(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    simplify_tolerance: float = typer.Option(2.0, "--simplify-tolerance", help="Geometric tolerance for simplification (meters)"),
    prune_epsilon: float = typer.Option(0.1, "--prune-epsilon", help="Threshold for pruning tiny segments (meters)"),
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Apply topology-preserving simplification to geometries."""
    try:
        config = CleanConfig(
            metric_crs=metric_crs,
            simplify_tolerance=simplify_tolerance,
            prune_epsilon=prune_epsilon,
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        console.print("Starting simplify operation...")
        gdf, report = run_simplify(input_path, output_path, config)
        
        # Display results
        vertices_removed = report['vertices_removed']
        vertex_reduction = report['vertex_reduction_percent']
        console.print(f"Reduced vertices: {report['vertices_before_simplify']} -> {report['vertices_after_simplify']} ({vertex_reduction:.1f}% reduction)")
        console.print(f"Pruned {report['pruned_segments']} tiny segments")
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features in {report['processing_time']:.2f}s")
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
        
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


@app.command("remove-components")
def remove_components_cmd(
    input_path: str = typer.Argument(..., help="Input Shapefile or GeoJSON path"),
    output_path: str = typer.Argument(..., help="Output GeoJSON path"),
    min_component_features: int = typer.Option(2, "--min-features", help="Minimum features per component to preserve"),
    min_component_length: float = typer.Option(10.0, "--min-length", help="Minimum total length (meters) per component to preserve"),
    metric_crs: str = typer.Option("auto", "--metric-crs", help="Metric CRS for processing"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
    strict_validation: bool = typer.Option(False, "--strict/--no-strict", help="Fail on validation issues"),
    report_path: Optional[str] = typer.Option(None, "--report", help="Save processing report to JSON file"),
) -> None:
    """Remove small isolated connected components from LineString network."""
    try:
        config = CleanConfig(
            metric_crs=metric_crs,
            remove_small_components=True,  # Enable component removal
            min_component_features=min_component_features,
            min_component_length=min_component_length,
            verbose=verbose,
            progress_bar=progress,
            strict_validation=strict_validation
        )
        
        console.print("Starting component removal...")
        gdf, report = run_remove_components(input_path, output_path, config)
        
        # Display results
        console.print(f"Found {report['small_components_found']} small components")
        console.print(f"Removed {report['small_components_removed']} components ({report['small_component_features_removed']} features)")
        console.print(f"Result: {report['output_connected_components']} connected component(s)")
        console.print(f"Processed {report['input_count']} -> {report['output_count']} features in {report['processing_time']:.2f}s")
        
        # Save report if requested
        if report_path:
            save_report(report, report_path)
            console.print(f"Report saved to {report_path}")
            
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose >= 2:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    app()