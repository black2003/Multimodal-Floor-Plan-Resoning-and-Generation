"""
Graph builder for floor plan representation
"""

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import networkx as nx
from ..preprocessing.segmentation import Room, Wall, Door

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Graph node representation"""
    id: int
    node_type: str  # 'room', 'door', 'wall'
    features: np.ndarray
    position: Tuple[float, float]
    metadata: Dict[str, Any] = None

@dataclass
class GraphEdge:
    """Graph edge representation"""
    source: int
    target: int
    edge_type: str  # 'adjacency', 'door_connection', 'wall_connection'
    features: np.ndarray
    weight: float = 1.0
    metadata: Dict[str, Any] = None

class GraphBuilder:
    """Build graph representation from segmented floor plan components"""
    
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # Room type embeddings
        self.room_type_embeddings = {
            'bedroom': [1, 0, 0, 0, 0, 0, 0, 0],
            'bathroom': [0, 1, 0, 0, 0, 0, 0, 0],
            'kitchen': [0, 0, 1, 0, 0, 0, 0, 0],
            'living_room': [0, 0, 0, 1, 0, 0, 0, 0],
            'dining_room': [0, 0, 0, 0, 1, 0, 0, 0],
            'closet': [0, 0, 0, 0, 0, 1, 0, 0],
            'hallway': [0, 0, 0, 0, 0, 0, 1, 0],
            'unknown': [0, 0, 0, 0, 0, 0, 0, 1]
        }
    
    def build_graph(self, segmentation_result: Dict[str, Any]) -> Data:
        """
        Build PyTorch Geometric graph from segmentation result
        
        Args:
            segmentation_result: Result from classical segmentation
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            rooms = segmentation_result['rooms']
            walls = segmentation_result['walls']
            doors = segmentation_result['doors']
            adjacency = segmentation_result['adjacency']
            
            # Create nodes
            nodes = self._create_nodes(rooms, walls, doors)
            
            # Create edges
            edges = self._create_edges(rooms, walls, doors, adjacency)
            
            # Convert to PyTorch Geometric format
            graph_data = self._to_pytorch_geometric(nodes, edges, segmentation_result)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise
    
    def _create_nodes(self, rooms: List[Room], walls: List[Wall], doors: List[Door]) -> List[GraphNode]:
        """Create graph nodes from segmented components"""
        try:
            nodes = []
            node_id = 0
            
            # Create room nodes
            for room in rooms:
                features = self._create_room_features(room)
                node = GraphNode(
                    id=node_id,
                    node_type='room',
                    features=features,
                    position=room.center,
                    metadata={
                        'room_id': room.id,
                        'area': room.area,
                        'room_type': room.room_type,
                        'bounding_box': room.bounding_box
                    }
                )
                nodes.append(node)
                node_id += 1
            
            # Create door nodes
            for door in doors:
                features = self._create_door_features(door)
                node = GraphNode(
                    id=node_id,
                    node_type='door',
                    features=features,
                    position=door.position,
                    metadata={
                        'door_id': door.id,
                        'width': door.width,
                        'orientation': door.orientation,
                        'room_connections': door.room_connections
                    }
                )
                nodes.append(node)
                node_id += 1
            
            # Create wall nodes (optional, for detailed analysis)
            for wall in walls:
                features = self._create_wall_features(wall)
                # Use midpoint of wall as position
                x1, y1 = wall.line[0]
                x2, y2 = wall.line[1]
                position = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                node = GraphNode(
                    id=node_id,
                    node_type='wall',
                    features=features,
                    position=position,
                    metadata={
                        'wall_id': wall.id,
                        'length': wall.length,
                        'thickness': wall.thickness,
                        'line': wall.line
                    }
                )
                nodes.append(node)
                node_id += 1
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error creating nodes: {str(e)}")
            return []
    
    def _create_room_features(self, room: Room) -> np.ndarray:
        """Create feature vector for room node"""
        try:
            features = np.zeros(self.node_feature_dim)
            
            # Basic geometric features
            features[0] = room.area / 10000  # Normalized area
            features[1] = room.center[0] / 1000  # Normalized x position
            features[2] = room.center[1] / 1000  # Normalized y position
            
            # Bounding box features
            if room.bounding_box:
                x, y, w, h = room.bounding_box
                features[3] = w / 1000  # Normalized width
                features[4] = h / 1000  # Normalized height
                features[5] = (w * h) / 1000000  # Normalized bounding box area
                features[6] = w / h if h > 0 else 1  # Aspect ratio
            
            # Room type embedding
            room_type_embedding = self.room_type_embeddings.get(room.room_type, 
                                                               self.room_type_embeddings['unknown'])
            features[7:15] = room_type_embedding
            
            # Contour features
            if hasattr(room, 'contour') and room.contour is not None:
                # Perimeter
                perimeter = cv2.arcLength(room.contour, True)
                features[15] = perimeter / 1000  # Normalized perimeter
                
                # Compactness (area/perimeter^2)
                if perimeter > 0:
                    features[16] = room.area / (perimeter * perimeter)
                
                # Convexity (area/convex_hull_area)
                hull = cv2.convexHull(room.contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    features[17] = room.area / hull_area
            
            # Fill remaining features with zeros or derived features
            # This ensures consistent feature dimension
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating room features: {str(e)}")
            return np.zeros(self.node_feature_dim)
    
    def _create_door_features(self, door: Door) -> np.ndarray:
        """Create feature vector for door node"""
        try:
            features = np.zeros(self.node_feature_dim)
            
            # Basic features
            features[0] = door.width / 100  # Normalized width
            features[1] = door.position[0] / 1000  # Normalized x position
            features[2] = door.position[1] / 1000  # Normalized y position
            features[3] = door.orientation / 180  # Normalized orientation
            
            # Door type indicator
            features[4] = 1.0  # Door node indicator
            
            # Room connection features
            if door.room_connections:
                features[5] = len(door.room_connections) / 10  # Number of connections
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating door features: {str(e)}")
            return np.zeros(self.node_feature_dim)
    
    def _create_wall_features(self, wall: Wall) -> np.ndarray:
        """Create feature vector for wall node"""
        try:
            features = np.zeros(self.node_feature_dim)
            
            # Basic features
            features[0] = wall.length / 1000  # Normalized length
            features[1] = wall.thickness / 100  # Normalized thickness
            
            # Position features (midpoint)
            x1, y1 = wall.line[0]
            x2, y2 = wall.line[1]
            features[2] = (x1 + x2) / 2000  # Normalized x position
            features[3] = (y1 + y2) / 2000  # Normalized y position
            
            # Orientation
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1)
                features[4] = angle / np.pi  # Normalized angle
            
            # Wall type indicator
            features[5] = 1.0  # Wall node indicator
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating wall features: {str(e)}")
            return np.zeros(self.node_feature_dim)
    
    def _create_edges(self, rooms: List[Room], walls: List[Wall], 
                     doors: List[Door], adjacency: Dict[int, List[int]]) -> List[GraphEdge]:
        """Create graph edges from component relationships"""
        try:
            edges = []
            edge_id = 0
            
            # Create adjacency edges between rooms
            for room_id, connected_rooms in adjacency.items():
                for connected_room_id in connected_rooms:
                    edge = GraphEdge(
                        source=room_id,
                        target=connected_room_id,
                        edge_type='adjacency',
                        features=self._create_adjacency_edge_features(rooms[room_id], rooms[connected_room_id]),
                        weight=1.0,
                        metadata={'connection_type': 'room_adjacency'}
                    )
                    edges.append(edge)
                    edge_id += 1
            
            # Create door connection edges
            for door in doors:
                if door.room_connections and len(door.room_connections) >= 2:
                    # Connect door to each room it connects
                    door_node_id = len(rooms) + door.id
                    
                    for room_id in door.room_connections:
                        if room_id < len(rooms):
                            edge = GraphEdge(
                                source=door_node_id,
                                target=room_id,
                                edge_type='door_connection',
                                features=self._create_door_edge_features(door, rooms[room_id]),
                                weight=1.0,
                                metadata={'connection_type': 'door_room'}
                            )
                            edges.append(edge)
                            edge_id += 1
            
            # Create wall connection edges (optional)
            for wall in walls:
                wall_node_id = len(rooms) + len(doors) + wall.id
                
                # Find rooms that this wall might separate
                for room in rooms:
                    if self._is_wall_adjacent_to_room(wall, room):
                        edge = GraphEdge(
                            source=wall_node_id,
                            target=room.id,
                            edge_type='wall_connection',
                            features=self._create_wall_edge_features(wall, room),
                            weight=0.5,  # Lower weight for wall connections
                            metadata={'connection_type': 'wall_room'}
                        )
                        edges.append(edge)
                        edge_id += 1
            
            return edges
            
        except Exception as e:
            logger.error(f"Error creating edges: {str(e)}")
            return []
    
    def _create_adjacency_edge_features(self, room1: Room, room2: Room) -> np.ndarray:
        """Create features for adjacency edge"""
        try:
            features = np.zeros(self.edge_feature_dim)
            
            # Distance between room centers
            dist = np.sqrt((room1.center[0] - room2.center[0])**2 + 
                          (room1.center[1] - room2.center[1])**2)
            features[0] = dist / 1000  # Normalized distance
            
            # Area ratio
            if room2.area > 0:
                features[1] = room1.area / room2.area
            
            # Room type compatibility
            if room1.room_type == room2.room_type:
                features[2] = 1.0
            
            # Edge type indicator
            features[3] = 1.0  # Adjacency edge indicator
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating adjacency edge features: {str(e)}")
            return np.zeros(self.edge_feature_dim)
    
    def _create_door_edge_features(self, door: Door, room: Room) -> np.ndarray:
        """Create features for door-room edge"""
        try:
            features = np.zeros(self.edge_feature_dim)
            
            # Distance from door to room center
            dist = np.sqrt((door.position[0] - room.center[0])**2 + 
                          (door.position[1] - room.center[1])**2)
            features[0] = dist / 1000  # Normalized distance
            
            # Door width relative to room size
            if room.area > 0:
                features[1] = door.width / np.sqrt(room.area)
            
            # Edge type indicator
            features[2] = 1.0  # Door connection indicator
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating door edge features: {str(e)}")
            return np.zeros(self.edge_feature_dim)
    
    def _create_wall_edge_features(self, wall: Wall, room: Room) -> np.ndarray:
        """Create features for wall-room edge"""
        try:
            features = np.zeros(self.edge_feature_dim)
            
            # Distance from wall midpoint to room center
            x1, y1 = wall.line[0]
            x2, y2 = wall.line[1]
            wall_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            dist = np.sqrt((wall_center[0] - room.center[0])**2 + 
                          (wall_center[1] - room.center[1])**2)
            features[0] = dist / 1000  # Normalized distance
            
            # Wall length relative to room size
            if room.area > 0:
                features[1] = wall.length / np.sqrt(room.area)
            
            # Edge type indicator
            features[2] = 1.0  # Wall connection indicator
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating wall edge features: {str(e)}")
            return np.zeros(self.edge_feature_dim)
    
    def _is_wall_adjacent_to_room(self, wall: Wall, room: Room) -> bool:
        """Check if wall is adjacent to room"""
        try:
            # Simple heuristic: check if wall is near room boundary
            x1, y1 = wall.line[0]
            x2, y2 = wall.line[1]
            wall_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if room.bounding_box:
                x, y, w, h = room.bounding_box
                # Check if wall center is near room boundary
                tolerance = 50
                return (abs(wall_center[0] - x) < tolerance or 
                        abs(wall_center[0] - (x + w)) < tolerance or
                        abs(wall_center[1] - y) < tolerance or 
                        abs(wall_center[1] - (y + h)) < tolerance)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking wall-room adjacency: {str(e)}")
            return False
    
    def _to_pytorch_geometric(self, nodes: List[GraphNode], edges: List[GraphEdge], 
                            segmentation_result: Dict[str, Any]) -> Data:
        """Convert to PyTorch Geometric Data format"""
        try:
            # Node features
            node_features = torch.stack([torch.tensor(node.features, dtype=torch.float32) 
                                       for node in nodes])
            
            # Edge indices
            edge_indices = torch.tensor([[edge.source, edge.target] for edge in edges], 
                                      dtype=torch.long).t().contiguous()
            
            # Edge features
            edge_features = torch.stack([torch.tensor(edge.features, dtype=torch.float32) 
                                       for edge in edges])
            
            # Edge weights
            edge_weights = torch.tensor([edge.weight for edge in edges], dtype=torch.float32)
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features,
                edge_index=edge_indices,
                edge_attr=edge_features,
                edge_weight=edge_weights,
                num_nodes=len(nodes)
            )
            
            # Add metadata
            data.node_types = [node.node_type for node in nodes]
            data.node_positions = torch.tensor([node.position for node in nodes], dtype=torch.float32)
            data.node_metadata = [node.metadata for node in nodes]
            data.edge_types = [edge.edge_type for edge in edges]
            data.edge_metadata = [edge.metadata for edge in edges]
            
            # Add segmentation metadata
            data.image_shape = torch.tensor(segmentation_result['image_shape'], dtype=torch.long)
            data.num_rooms = len(segmentation_result['rooms'])
            data.num_doors = len(segmentation_result['doors'])
            data.num_walls = len(segmentation_result['walls'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error converting to PyTorch Geometric: {str(e)}")
            raise
    
    def visualize_graph(self, graph_data: Data, save_path: Optional[str] = None) -> None:
        """Visualize the graph structure"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for i, node_type in enumerate(graph_data.node_types):
                G.add_node(i, node_type=node_type)
            
            # Add edges
            edge_index = graph_data.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                source, target = edge_index[:, i]
                G.add_edge(int(source), int(target))
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Color nodes by type
            node_colors = []
            for node_type in graph_data.node_types:
                if node_type == 'room':
                    node_colors.append('lightblue')
                elif node_type == 'door':
                    node_colors.append('red')
                elif node_type == 'wall':
                    node_colors.append('gray')
                else:
                    node_colors.append('white')
            
            # Draw graph
            nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                   node_size=500, font_size=8, font_weight='bold')
            
            plt.title('Floor Plan Graph Structure')
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            # Don't raise exception for visualization errors
