"""Core data types and configuration for tnclean."""

from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Literal, Optional, Tuple
import hashlib

# Type aliases for cleaner signatures
PrecisionMode = Literal["grid", "decimal"]
SimplifyMode = Literal["topology", "rdp", "vw"]
DedupPolicy = Literal["keep-shortest", "keep-first", "keep-longest"]


@dataclass
class CleanConfig:
    """Configuration for the tnclean pipeline.
    
    All distance tolerances are in meters when using metric CRS.
    """
    
    # CRS and projection
    metric_crs: Literal["auto"] | str = "auto"
    crs: Optional[str] = None
    
    # Core tolerances (meters) 
    snap_precision: float | Literal["auto"] = "auto"
    overlap_tolerance: float = 0.5
    simplify_tolerance: float = 2.0
    prune_epsilon: float = 0.1
    
    # Processing modes
    snap_mode: PrecisionMode = "grid"
    simplify_mode: SimplifyMode = "topology"
    dedup_policy: DedupPolicy = "keep-shortest"
    keep_parallel_edges: bool = False
    check_geometry_paths: bool = False  # Enable path-based deduplication instead of endpoint-only
    
    # Degree-2 node removal
    merge_degree2_nodes: bool = True  # Enable merging of degree-2 nodes
    merge_attribute_policy: Literal["concatenate", "keep-longest", "keep-first"] = "concatenate"
    merge_max_length_ratio: float = 10.0  # Max ratio between merged segments (longer/shorter)
    
    # Small component removal
    remove_small_components: bool = True   # Enable removal of small isolated components
    min_component_features: int = 1        # Minimum features per component to preserve
    min_component_length: float = 5.0      # Minimum total length (meters) per component to preserve
    
    # Pairing performance filters
    pairing_batch_size: int = 100_000
    pairing_bearing_delta_deg: float = 5.0
    pairing_length_ratio_min: float = 0.5
    pairing_length_ratio_max: float = 2.0
    pairing_overlap_alpha: float = 0.9
    
    # Field handling
    preserve_fields: Optional[Iterable[str]] = None
    dissolve_fields: Optional[Iterable[str]] = None
    
    # Processing options
    n_workers: int = 0
    strict_validation: bool = False
    progress_bar: bool = True
    verbose: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if isinstance(self.snap_precision, (int, float)) and self.snap_precision <= 0:
            raise ValueError("snap_precision must be positive")
        if self.overlap_tolerance <= 0:
            raise ValueError("overlap_tolerance must be positive")
        if self.simplify_tolerance < 0:
            raise ValueError("simplify_tolerance must be non-negative")
        if self.prune_epsilon < 0:
            raise ValueError("prune_epsilon must be non-negative")
        if not (0 <= self.pairing_length_ratio_min <= self.pairing_length_ratio_max):
            raise ValueError("Invalid pairing length ratio bounds")
        if not (0 <= self.pairing_overlap_alpha <= 1):
            raise ValueError("pairing_overlap_alpha must be in [0,1]")


@dataclass(frozen=True)
class NodeId:
    """Quantized-Coordinate Key (QCK) for node identity.
    
    Represents a node by its quantized coordinates in a specific CRS.
    """
    ix: int  # quantized x coordinate
    iy: int  # quantized y coordinate
    crs: str  # coordinate reference system
    precision: float  # snap precision used
    
    def __str__(self) -> str:
        return f"{self.ix}:{self.iy}:{self.crs}:{self.precision}"
    
    def __hash__(self) -> int:
        """Efficient hash for use in sets/dicts."""
        return hash((self.ix, self.iy, self.crs, self.precision))


@dataclass(frozen=True)
class EdgeKey:
    """Unordered edge key for deduplication.
    
    Represents an edge between two nodes, order-independent.
    """
    node_a: NodeId
    node_b: NodeId
    
    def __post_init__(self) -> None:
        """Ensure canonical ordering for consistent hashing."""
        # Sort nodes to ensure consistent ordering
        if (self.node_a.ix, self.node_a.iy) > (self.node_b.ix, self.node_b.iy):
            # Swap the nodes - need to store original node_a before overwriting
            original_node_a = self.node_a
            object.__setattr__(self, 'node_a', self.node_b)
            object.__setattr__(self, 'node_b', original_node_a)
    
    def __hash__(self) -> int:
        return hash((self.node_a, self.node_b))
    
    def __str__(self) -> str:
        return f"{self.node_a} <-> {self.node_b}"


@dataclass
class ProcessingStats:
    """Statistics collected during processing."""
    
    # Input statistics
    input_count: int = 0
    input_multilinestring_count: int = 0
    input_total_length: float = 0.0
    
    # Processing statistics
    exploded_count: int = 0
    snapped_vertex_changes: int = 0
    split_count: int = 0
    overlap_pairs_found: int = 0
    duplicates_removed: int = 0
    
    # Simplification statistics
    vertices_before_simplify: int = 0
    vertices_after_simplify: int = 0
    pruned_segments: int = 0
    
    # Degree-2 node removal statistics
    degree2_nodes_found: int = 0
    degree2_merges_performed: int = 0
    
    # Small component removal statistics
    small_components_found: int = 0
    small_components_removed: int = 0
    small_component_features_removed: int = 0
    
    # Output statistics
    output_count: int = 0
    output_total_length: float = 0.0
    output_connected_components: int = 0
    
    # Performance metrics
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0


@dataclass
class ValidationBounds:
    """Data-driven bounds for tolerance validation."""
    
    # Dataset statistics
    vertex_spacing_p05: float
    vertex_spacing_p50: float
    median_edge_length: float
    dataset_diagonal: float
    
    # Derived bounds
    snap_precision_min: float
    snap_precision_max: float
    overlap_tolerance_max: float
    simplify_tolerance_max: float
    prune_epsilon_max: float
    
    @classmethod
    def compute_bounds(
        cls, 
        vertex_spacings: list[float], 
        edge_lengths: list[float],
        bbox_diagonal: float
    ) -> "ValidationBounds":
        """Compute validation bounds from dataset statistics."""
        import numpy as np
        
        v05 = np.percentile(vertex_spacings, 5)
        v50 = np.percentile(vertex_spacings, 50)
        l50 = np.median(edge_lengths)
        
        return cls(
            vertex_spacing_p05=v05,
            vertex_spacing_p50=v50,
            median_edge_length=l50,
            dataset_diagonal=bbox_diagonal,
            snap_precision_min=0.25 * v05,
            snap_precision_max=0.75 * v50,
            overlap_tolerance_max=0.5,  # relative to snap_precision
            simplify_tolerance_max=min(2 * l50, 0.05 * bbox_diagonal),
            prune_epsilon_max=0.1 * v50,
        )
    
    def validate_config(self, config: CleanConfig, strict: bool = False) -> list[str]:
        """Validate config against bounds, return list of warnings/errors."""
        issues = []
        
        # Validate snap_precision (only if not "auto")
        if isinstance(config.snap_precision, (int, float)):
            if not (self.snap_precision_min <= config.snap_precision <= self.snap_precision_max):
                msg = (f"snap_precision {config.snap_precision}m outside recommended range "
                       f"[{self.snap_precision_min:.3f}, {self.snap_precision_max:.3f}]m")
                issues.append(msg)
        
        # Validate overlap_tolerance (only if snap_precision is numeric)
        if isinstance(config.snap_precision, (int, float)):
            max_overlap = self.overlap_tolerance_max * config.snap_precision
            if config.overlap_tolerance > max_overlap:
                msg = (f"overlap_tolerance {config.overlap_tolerance} > "
                       f"{self.overlap_tolerance_max}×snap_precision = {max_overlap:.3f}")
                issues.append(msg)
        
        # Validate simplify_tolerance
        if config.simplify_tolerance > self.simplify_tolerance_max:
            msg = (f"simplify_tolerance {config.simplify_tolerance} > "
                   f"recommended max {self.simplify_tolerance_max:.3f}")
            issues.append(msg)
        
        # Validate prune_epsilon
        max_prune = min(
            0.5 * config.simplify_tolerance,
            config.overlap_tolerance,
            self.prune_epsilon_max
        )
        if config.prune_epsilon > max_prune:
            msg = (f"prune_epsilon {config.prune_epsilon} > "
                   f"recommended max {max_prune:.3f}")
            issues.append(msg)
        
        return issues
    
    def clamp_config(self, config: CleanConfig) -> CleanConfig:
        """Return clamped config with values within bounds."""
        from copy import deepcopy
        
        new_config = deepcopy(config)
        
        # Set snap_precision: auto -> use 5th percentile, numeric -> clamp if needed
        if config.snap_precision == "auto":
            new_config.snap_precision = self.snap_precision_min  # Use V05 as default
        elif isinstance(config.snap_precision, (int, float)):
            # Keep user-specified values, don't clamp
            new_config.snap_precision = config.snap_precision
        else:
            # Fallback for invalid types
            new_config.snap_precision = self.snap_precision_min
        
        # Clamp overlap_tolerance
        max_overlap = self.overlap_tolerance_max * new_config.snap_precision
        new_config.overlap_tolerance = min(max_overlap, config.overlap_tolerance)
        
        # Clamp simplify_tolerance
        new_config.simplify_tolerance = min(
            self.simplify_tolerance_max, config.simplify_tolerance
        )
        new_config.simplify_tolerance = max(
            new_config.snap_precision, new_config.simplify_tolerance
        )
        
        # Clamp prune_epsilon
        max_prune = min(
            0.5 * new_config.simplify_tolerance,
            new_config.overlap_tolerance,
            self.prune_epsilon_max
        )
        new_config.prune_epsilon = min(max_prune, config.prune_epsilon)
        
        return new_config