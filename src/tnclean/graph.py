"""Graph utilities and QCK (Quantized-Coordinate Key) node identity system."""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from shapely.geometry import Point, LineString
from shapely import get_coordinates
import geopandas as gpd

from .types import NodeId, EdgeKey, CleanConfig


class QCKGraph:
    """Graph with Quantized-Coordinate Key (QCK) node identity.
    
    Manages nodes and edges using quantized coordinates for stable topology.
    """
    
    def __init__(self, config: CleanConfig, crs: str):
        self.config = config
        self.crs = crs
        self._nodes: Dict[NodeId, Point] = {}
        self._edges: Dict[EdgeKey, List[int]] = {}  # EdgeKey -> list of feature indices
        self._edge_geometries: Dict[EdgeKey, LineString] = {}
        self._edge_attributes: Dict[EdgeKey, Dict] = {}
    
    def quantize_point(self, x: float, y: float) -> NodeId:
        """Quantize coordinates to create stable node identity."""
        precision = self.config.snap_precision
        ix = int(np.floor(x / precision))
        iy = int(np.floor(y / precision))
        return NodeId(ix=ix, iy=iy, crs=self.crs, precision=precision)
    
    def snap_coordinate(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinate to quantized grid."""
        node_id = self.quantize_point(x, y)
        precision = self.config.snap_precision
        snapped_x = (node_id.ix + 0.5) * precision
        snapped_y = (node_id.iy + 0.5) * precision
        return snapped_x, snapped_y
    
    def add_node(self, x: float, y: float) -> NodeId:
        """Add or retrieve node with quantized coordinates."""
        snapped_x, snapped_y = self.snap_coordinate(x, y)
        node_id = self.quantize_point(x, y)
        
        if node_id not in self._nodes:
            self._nodes[node_id] = Point(snapped_x, snapped_y)
        
        return node_id
    
    def add_edge(
        self, 
        linestring: LineString, 
        feature_idx: int,
        attributes: Optional[Dict] = None
    ) -> EdgeKey:
        """Add edge to graph with QCK endpoints."""
        coords = get_coordinates(linestring)
        
        # Create nodes for endpoints
        start_node = self.add_node(coords[0, 0], coords[0, 1])
        end_node = self.add_node(coords[-1, 0], coords[-1, 1])
        
        # Create edge key
        edge_key = EdgeKey(start_node, end_node)
        
        # Store edge information
        if edge_key not in self._edges:
            self._edges[edge_key] = []
            self._edge_geometries[edge_key] = linestring
            self._edge_attributes[edge_key] = attributes or {}
        else:
            # Add to existing edge (for deduplication handling)
            self._edges[edge_key].append(feature_idx)
        
        if feature_idx not in self._edges[edge_key]:
            self._edges[edge_key].append(feature_idx)
        
        return edge_key
    
    def get_node_coordinates(self, node_id: NodeId) -> Tuple[float, float]:
        """Get actual coordinates for a node."""
        point = self._nodes[node_id]
        return point.x, point.y
    
    def get_edge_endpoints(self, edge_key: EdgeKey) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get endpoint coordinates for an edge."""
        start_coords = self.get_node_coordinates(edge_key.node_a)
        end_coords = self.get_node_coordinates(edge_key.node_b)
        return start_coords, end_coords
    
    def snap_linestring(self, linestring: LineString) -> LineString:
        """Snap all vertices of a linestring to the quantized grid."""
        coords = get_coordinates(linestring)
        snapped_coords = []
        
        for x, y in coords:
            snapped_x, snapped_y = self.snap_coordinate(x, y)
            snapped_coords.append((snapped_x, snapped_y))
        
        return LineString(snapped_coords)
    
    def find_duplicates(self) -> Dict[EdgeKey, List[int]]:
        """Find edges with duplicate endpoints."""
        duplicates = {}
        for edge_key, feature_indices in self._edges.items():
            if len(feature_indices) > 1:
                duplicates[edge_key] = feature_indices
        return duplicates
    
    def get_node_degree(self, node_id: NodeId) -> int:
        """Count how many edges are incident to a node."""
        degree = 0
        for edge_key in self._edges:
            if node_id in (edge_key.node_a, edge_key.node_b):
                degree += 1
        return degree
    
    def get_incident_edges(self, node_id: NodeId) -> List[EdgeKey]:
        """Get all edges incident to a node."""
        incident = []
        for edge_key in self._edges:
            if node_id in (edge_key.node_a, edge_key.node_b):
                incident.append(edge_key)
        return incident
    
    def export_nodes_gdf(self) -> gpd.GeoDataFrame:
        """Export nodes as a GeoDataFrame."""
        if not self._nodes:
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)
        
        data = []
        for node_id, point in self._nodes.items():
            data.append({
                'node_id': str(node_id),
                'ix': node_id.ix,
                'iy': node_id.iy,
                'precision': node_id.precision,
                'degree': self.get_node_degree(node_id),
                'geometry': point
            })
        
        return gpd.GeoDataFrame(data, crs=self.crs)
    
    def export_edges_gdf(self) -> gpd.GeoDataFrame:
        """Export edges as a GeoDataFrame."""
        if not self._edges:
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)
        
        data = []
        for edge_key, feature_indices in self._edges.items():
            geometry = self._edge_geometries.get(edge_key)
            attributes = self._edge_attributes.get(edge_key, {})
            
            record = {
                'edge_key': str(edge_key),
                'node_a': str(edge_key.node_a),
                'node_b': str(edge_key.node_b),
                'feature_count': len(feature_indices),
                'feature_indices': feature_indices,
                'geometry': geometry,
                **attributes
            }
            data.append(record)
        
        return gpd.GeoDataFrame(data, crs=self.crs)


def compute_dataset_stats(gdf: gpd.GeoDataFrame) -> Tuple[List[float], List[float], float]:
    """Compute dataset statistics for tolerance validation.
    
    Returns:
        vertex_spacings: List of distances between consecutive vertices
        edge_lengths: List of edge lengths  
        bbox_diagonal: Diagonal of dataset bounding box
    """
    vertex_spacings = []
    edge_lengths = []
    
    for geometry in gdf.geometry:
        if geometry.geom_type == 'LineString':
            coords = get_coordinates(geometry)
            edge_lengths.append(geometry.length)
            
            # Compute vertex spacings
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                spacing = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if spacing > 0:  # Skip zero-length segments
                    vertex_spacings.append(spacing)
        
        elif geometry.geom_type == 'MultiLineString':
            for linestring in geometry.geoms:
                coords = get_coordinates(linestring)
                edge_lengths.append(linestring.length)
                
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    spacing = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if spacing > 0:
                        vertex_spacings.append(spacing)
    
    # Compute bounding box diagonal
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    bbox_diagonal = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    
    return vertex_spacings, edge_lengths, bbox_diagonal


def analyze_network_topology(gdf: gpd.GeoDataFrame, config: CleanConfig, crs: str) -> Dict:
    """Analyze network topology and return diagnostic information."""
    graph = QCKGraph(config, crs)
    
    # Add all edges to graph
    for idx, row in gdf.iterrows():
        if row.geometry.geom_type == 'LineString':
            graph.add_edge(row.geometry, idx, dict(row.drop('geometry')))
        elif row.geometry.geom_type == 'MultiLineString':
            for i, linestring in enumerate(row.geometry.geoms):
                # Use negative indices for sub-parts of multilinestrings
                sub_idx = -(idx * 1000 + i + 1)
                graph.add_edge(linestring, sub_idx, dict(row.drop('geometry')))
    
    # Analyze topology
    duplicates = graph.find_duplicates()
    node_degrees = {nid: graph.get_node_degree(nid) for nid in graph._nodes}
    
    return {
        'total_nodes': len(graph._nodes),
        'total_edges': len(graph._edges),
        'duplicate_edge_keys': len(duplicates),
        'total_duplicate_features': sum(len(indices) for indices in duplicates.values()),
        'node_degree_distribution': {
            'degree_1': sum(1 for d in node_degrees.values() if d == 1),
            'degree_2': sum(1 for d in node_degrees.values() if d == 2), 
            'degree_3_plus': sum(1 for d in node_degrees.values() if d >= 3),
        },
        'max_node_degree': max(node_degrees.values()) if node_degrees else 0,
    }