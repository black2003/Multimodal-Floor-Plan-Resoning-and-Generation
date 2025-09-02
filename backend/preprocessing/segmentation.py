"""
Classical computer vision segmentation for floor plans
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Room:
    """Room data structure"""
    id: int
    contour: np.ndarray
    area: float
    center: Tuple[float, float]
    room_type: str = "unknown"
    bounding_box: Tuple[int, int, int, int] = None

@dataclass
class Wall:
    """Wall data structure"""
    id: int
    line: Tuple[Tuple[int, int], Tuple[int, int]]
    length: float
    thickness: float = 5.0

@dataclass
class Door:
    """Door data structure"""
    id: int
    position: Tuple[int, int]
    width: float
    orientation: float
    room_connections: List[int] = None

class ClassicalSegmentation:
    """Classical computer vision segmentation for floor plans"""
    
    def __init__(self):
        self.min_room_area = 1000  # Minimum area for a room
        self.max_wall_thickness = 20  # Maximum wall thickness
        self.door_width_range = (30, 120)  # Door width range in pixels
        
    def segment_floorplan(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Segment floor plan into rooms, walls, and doors
        
        Args:
            image: Input floor plan image
            
        Returns:
            Dictionary containing segmented components
        """
        try:
            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Segment different components
            rooms = self._segment_rooms(processed)
            walls = self._segment_walls(processed)
            doors = self._segment_doors(processed, walls)
            
            # Build adjacency graph
            adjacency = self._build_adjacency_graph(rooms, doors)
            
            return {
                'rooms': rooms,
                'walls': walls,
                'doors': doors,
                'adjacency': adjacency,
                'image_shape': image.shape
            }
            
        except Exception as e:
            logger.error(f"Error segmenting floor plan: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for segmentation"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Invert image (walls should be white, rooms black)
            inverted = cv2.bitwise_not(thresh)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image
    
    def _segment_rooms(self, processed_image: np.ndarray) -> List[Room]:
        """Segment rooms from processed image"""
        try:
            rooms = []
            
            # Find contours
            contours, _ = cv2.findContours(
                processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and process contours
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.min_room_area:
                    continue
                
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create room object
                room = Room(
                    id=i,
                    contour=contour,
                    area=area,
                    center=(cx, cy),
                    bounding_box=(x, y, w, h)
                )
                
                rooms.append(room)
            
            return rooms
            
        except Exception as e:
            logger.error(f"Error segmenting rooms: {str(e)}")
            return []
    
    def _segment_walls(self, processed_image: np.ndarray) -> List[Wall]:
        """Segment walls from processed image"""
        try:
            walls = []
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                processed_image, 1, np.pi/180, threshold=100,
                minLineLength=50, maxLineGap=10
            )
            
            if lines is not None:
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Filter by length
                    if length < 30:
                        continue
                    
                    # Create wall object
                    wall = Wall(
                        id=i,
                        line=((x1, y1), (x2, y2)),
                        length=length
                    )
                    
                    walls.append(wall)
            
            return walls
            
        except Exception as e:
            logger.error(f"Error segmenting walls: {str(e)}")
            return []
    
    def _segment_doors(self, processed_image: np.ndarray, walls: List[Wall]) -> List[Door]:
        """Segment doors from processed image and walls"""
        try:
            doors = []
            
            # Create a copy of the image for door detection
            door_image = processed_image.copy()
            
            # Remove walls to isolate potential door areas
            for wall in walls:
                x1, y1 = wall.line[0]
                x2, y2 = wall.line[1]
                cv2.line(door_image, (x1, y1), (x2, y2), 0, self.max_wall_thickness)
            
            # Find contours in door areas
            contours, _ = cv2.findContours(
                door_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(contours):
                # Calculate contour properties
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by area and aspect ratio (doors are typically rectangular)
                aspect_ratio = w / h if h > 0 else 0
                
                if (area > 100 and area < 2000 and 
                    aspect_ratio > 0.5 and aspect_ratio < 3.0):
                    
                    # Calculate center position
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Determine orientation
                    orientation = 0 if w > h else 90
                    
                    # Create door object
                    door = Door(
                        id=i,
                        position=(center_x, center_y),
                        width=max(w, h),
                        orientation=orientation
                    )
                    
                    doors.append(door)
            
            return doors
            
        except Exception as e:
            logger.error(f"Error segmenting doors: {str(e)}")
            return []
    
    def _build_adjacency_graph(self, rooms: List[Room], doors: List[Door]) -> Dict[int, List[int]]:
        """Build adjacency graph based on room connections through doors"""
        try:
            adjacency = {room.id: [] for room in rooms}
            
            # For each door, find which rooms it connects
            for door in doors:
                connected_rooms = []
                
                for room in rooms:
                    # Check if door is near room boundary
                    if self._is_door_connected_to_room(door, room):
                        connected_rooms.append(room.id)
                
                # Add connections to adjacency graph
                if len(connected_rooms) >= 2:
                    for i in range(len(connected_rooms)):
                        for j in range(i + 1, len(connected_rooms)):
                            room1_id = connected_rooms[i]
                            room2_id = connected_rooms[j]
                            
                            if room2_id not in adjacency[room1_id]:
                                adjacency[room1_id].append(room2_id)
                            if room1_id not in adjacency[room2_id]:
                                adjacency[room2_id].append(room1_id)
            
            return adjacency
            
        except Exception as e:
            logger.error(f"Error building adjacency graph: {str(e)}")
            return {}
    
    def _is_door_connected_to_room(self, door: Door, room: Room) -> bool:
        """Check if a door is connected to a room"""
        try:
            door_x, door_y = door.position
            x, y, w, h = room.bounding_box
            
            # Check if door is within room boundary (with some tolerance)
            tolerance = 20
            return (x - tolerance <= door_x <= x + w + tolerance and 
                    y - tolerance <= door_y <= y + h + tolerance)
            
        except Exception as e:
            logger.error(f"Error checking door-room connection: {str(e)}")
            return False
    
    def classify_room_types(self, rooms: List[Room], image: np.ndarray) -> List[Room]:
        """Classify room types based on size and position"""
        try:
            for room in rooms:
                # Simple classification based on area and position
                area = room.area
                center_x, center_y = room.center
                height, width = image.shape[:2]
                
                # Normalize position
                norm_x = center_x / width
                norm_y = center_y / height
                
                # Classify based on heuristics
                if area > 50000:  # Large area
                    if norm_x < 0.3:
                        room.room_type = "living_room"
                    elif norm_x > 0.7:
                        room.room_type = "bedroom"
                    else:
                        room.room_type = "kitchen"
                elif area > 20000:  # Medium area
                    if norm_y < 0.3:
                        room.room_type = "bathroom"
                    else:
                        room.room_type = "bedroom"
                else:  # Small area
                    room.room_type = "closet"
            
            return rooms
            
        except Exception as e:
            logger.error(f"Error classifying room types: {str(e)}")
            return rooms
