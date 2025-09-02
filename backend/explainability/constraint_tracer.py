"""
Constraint tracing for neuro-symbolic validation
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Types of architectural constraints"""
    ROOM_CONNECTIVITY = "room_connectivity"
    DOOR_PLACEMENT = "door_placement"
    ROOM_SIZE = "room_size"
    ACCESSIBILITY = "accessibility"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    CIRCULATION = "circulation"
    CODE_COMPLIANCE = "code_compliance"

class ViolationSeverity(Enum):
    """Severity levels for constraint violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConstraintViolation:
    """Represents a constraint violation"""
    constraint_type: ConstraintType
    severity: ViolationSeverity
    description: str
    affected_elements: List[str]
    suggested_fix: str
    confidence: float
    rule_id: str

@dataclass
class ConstraintRule:
    """Represents a constraint rule"""
    rule_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    check_function: callable
    severity: ViolationSeverity
    weight: float = 1.0

class ConstraintTracer:
    """Tracer for architectural constraint violations"""
    
    def __init__(self):
        self.constraint_rules = self._initialize_constraint_rules()
        self.violation_history = []
        
    def _initialize_constraint_rules(self) -> List[ConstraintRule]:
        """Initialize architectural constraint rules"""
        rules = []
        
        # Room connectivity rules
        rules.append(ConstraintRule(
            rule_id="RC001",
            name="No Isolated Rooms",
            description="All rooms must be accessible from at least one other room",
            constraint_type=ConstraintType.ROOM_CONNECTIVITY,
            check_function=self._check_room_connectivity,
            severity=ViolationSeverity.HIGH,
            weight=2.0
        ))
        
        rules.append(ConstraintRule(
            rule_id="RC002",
            name="Minimum Room Connections",
            description="Each room should have at least one door",
            constraint_type=ConstraintType.ROOM_CONNECTIVITY,
            check_function=self._check_minimum_connections,
            severity=ViolationSeverity.MEDIUM,
            weight=1.5
        ))
        
        # Door placement rules
        rules.append(ConstraintRule(
            rule_id="DP001",
            name="Proper Door Placement",
            description="Doors should be placed on walls, not in corners",
            constraint_type=ConstraintType.DOOR_PLACEMENT,
            check_function=self._check_door_placement,
            severity=ViolationSeverity.MEDIUM,
            weight=1.0
        ))
        
        rules.append(ConstraintRule(
            rule_id="DP002",
            name="Door Width Standards",
            description="Doors should meet minimum width requirements",
            constraint_type=ConstraintType.DOOR_PLACEMENT,
            check_function=self._check_door_width,
            severity=ViolationSeverity.LOW,
            weight=0.8
        ))
        
        # Room size rules
        rules.append(ConstraintRule(
            rule_id="RS001",
            name="Minimum Room Size",
            description="Rooms should meet minimum size requirements",
            constraint_type=ConstraintType.ROOM_SIZE,
            check_function=self._check_room_size,
            severity=ViolationSeverity.MEDIUM,
            weight=1.2
        ))
        
        rules.append(ConstraintRule(
            rule_id="RS002",
            name="Proportional Room Sizes",
            description="Room sizes should be proportional to their function",
            constraint_type=ConstraintType.ROOM_SIZE,
            check_function=self._check_room_proportions,
            severity=ViolationSeverity.LOW,
            weight=0.7
        ))
        
        # Accessibility rules
        rules.append(ConstraintRule(
            rule_id="AC001",
            name="Accessibility Path",
            description="All rooms should be accessible from the main entrance",
            constraint_type=ConstraintType.ACCESSIBILITY,
            check_function=self._check_accessibility,
            severity=ViolationSeverity.HIGH,
            weight=2.5
        ))
        
        rules.append(ConstraintRule(
            rule_id="AC002",
            name="Wheelchair Accessibility",
            description="Pathways should be wide enough for wheelchair access",
            constraint_type=ConstraintType.ACCESSIBILITY,
            check_function=self._check_wheelchair_access,
            severity=ViolationSeverity.MEDIUM,
            weight=1.8
        ))
        
        # Structural integrity rules
        rules.append(ConstraintRule(
            rule_id="SI001",
            name="Load Bearing Walls",
            description="Load bearing walls should not be removed without support",
            constraint_type=ConstraintType.STRUCTURAL_INTEGRITY,
            check_function=self._check_structural_integrity,
            severity=ViolationSeverity.CRITICAL,
            weight=3.0
        ))
        
        # Circulation rules
        rules.append(ConstraintRule(
            rule_id="CI001",
            name="Efficient Circulation",
            description="Floor plan should have efficient circulation patterns",
            constraint_type=ConstraintType.CIRCULATION,
            check_function=self._check_circulation,
            severity=ViolationSeverity.MEDIUM,
            weight=1.3
        ))
        
        return rules
    
    def trace_constraints(self, graph_data, segmentation_result: Dict[str, Any], 
                         neural_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Trace constraint violations in the floor plan
        
        Args:
            graph_data: PyTorch Geometric Data object
            segmentation_result: Result from classical segmentation
            neural_scores: Neural network confidence scores
            
        Returns:
            Dictionary containing constraint analysis results
        """
        try:
            violations = []
            rule_traces = []
            
            # Check each constraint rule
            for rule in self.constraint_rules:
                try:
                    # Run the constraint check
                    violation_result = rule.check_function(
                        graph_data, segmentation_result, neural_scores
                    )
                    
                    if violation_result:
                        violations.append(violation_result)
                    
                    # Create rule trace
                    rule_trace = {
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'constraint_type': rule.constraint_type.value,
                        'severity': rule.severity.value,
                        'weight': rule.weight,
                        'violated': violation_result is not None,
                        'violation': violation_result.__dict__ if violation_result else None
                    }
                    rule_traces.append(rule_trace)
                    
                except Exception as e:
                    logger.error(f"Error checking rule {rule.rule_id}: {str(e)}")
                    continue
            
            # Calculate overall validity score
            validity_score = self._calculate_validity_score(violations, rule_traces)
            
            # Generate repair suggestions
            repair_suggestions = self._generate_repair_suggestions(violations)
            
            # Store in history
            self.violation_history.append({
                'violations': violations,
                'rule_traces': rule_traces,
                'validity_score': validity_score,
                'timestamp': self._get_timestamp()
            })
            
            return {
                'violations': [v.__dict__ for v in violations],
                'rule_traces': rule_traces,
                'validity_score': validity_score,
                'repair_suggestions': repair_suggestions,
                'summary': self._create_summary(violations, validity_score)
            }
            
        except Exception as e:
            logger.error(f"Error tracing constraints: {str(e)}")
            return {}
    
    def _check_room_connectivity(self, graph_data, segmentation_result: Dict[str, Any], 
                               neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check if all rooms are properly connected"""
        try:
            # Get adjacency information
            adjacency = segmentation_result.get('adjacency', {})
            rooms = segmentation_result.get('rooms', [])
            
            isolated_rooms = []
            for room_id, connected_rooms in adjacency.items():
                if not connected_rooms:  # No connections
                    isolated_rooms.append(f"room_{room_id}")
            
            if isolated_rooms:
                return ConstraintViolation(
                    constraint_type=ConstraintType.ROOM_CONNECTIVITY,
                    severity=ViolationSeverity.HIGH,
                    description=f"Isolated rooms found: {', '.join(isolated_rooms)}",
                    affected_elements=isolated_rooms,
                    suggested_fix="Add doors or corridors to connect isolated rooms",
                    confidence=0.9,
                    rule_id="RC001"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking room connectivity: {str(e)}")
            return None
    
    def _check_minimum_connections(self, graph_data, segmentation_result: Dict[str, Any], 
                                 neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check minimum connection requirements"""
        try:
            doors = segmentation_result.get('doors', [])
            rooms = segmentation_result.get('rooms', [])
            
            # Check if each room has at least one door
            rooms_without_doors = []
            for room in rooms:
                has_door = any(
                    door.room_connections and room.id in door.room_connections 
                    for door in doors
                )
                if not has_door:
                    rooms_without_doors.append(f"room_{room.id}")
            
            if rooms_without_doors:
                return ConstraintViolation(
                    constraint_type=ConstraintType.ROOM_CONNECTIVITY,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Rooms without doors: {', '.join(rooms_without_doors)}",
                    affected_elements=rooms_without_doors,
                    suggested_fix="Add doors to rooms that lack access",
                    confidence=0.8,
                    rule_id="RC002"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking minimum connections: {str(e)}")
            return None
    
    def _check_door_placement(self, graph_data, segmentation_result: Dict[str, Any], 
                            neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check door placement constraints"""
        try:
            doors = segmentation_result.get('doors', [])
            walls = segmentation_result.get('walls', [])
            
            misplaced_doors = []
            for door in doors:
                # Check if door is near a wall
                door_x, door_y = door.position
                near_wall = False
                
                for wall in walls:
                    x1, y1 = wall.line[0]
                    x2, y2 = wall.line[1]
                    
                    # Calculate distance from door to wall line
                    distance = self._point_to_line_distance(door_x, door_y, x1, y1, x2, y2)
                    if distance < 20:  # Within 20 pixels
                        near_wall = True
                        break
                
                if not near_wall:
                    misplaced_doors.append(f"door_{door.id}")
            
            if misplaced_doors:
                return ConstraintViolation(
                    constraint_type=ConstraintType.DOOR_PLACEMENT,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Misplaced doors: {', '.join(misplaced_doors)}",
                    affected_elements=misplaced_doors,
                    suggested_fix="Reposition doors to be on walls",
                    confidence=0.7,
                    rule_id="DP001"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking door placement: {str(e)}")
            return None
    
    def _check_door_width(self, graph_data, segmentation_result: Dict[str, Any], 
                         neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check door width standards"""
        try:
            doors = segmentation_result.get('doors', [])
            narrow_doors = []
            
            min_width = 30  # Minimum door width in pixels
            
            for door in doors:
                if door.width < min_width:
                    narrow_doors.append(f"door_{door.id}")
            
            if narrow_doors:
                return ConstraintViolation(
                    constraint_type=ConstraintType.DOOR_PLACEMENT,
                    severity=ViolationSeverity.LOW,
                    description=f"Narrow doors: {', '.join(narrow_doors)}",
                    affected_elements=narrow_doors,
                    suggested_fix="Increase door width to meet standards",
                    confidence=0.6,
                    rule_id="DP002"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking door width: {str(e)}")
            return None
    
    def _check_room_size(self, graph_data, segmentation_result: Dict[str, Any], 
                        neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check room size constraints"""
        try:
            rooms = segmentation_result.get('rooms', [])
            undersized_rooms = []
            
            min_area = 1000  # Minimum room area in pixels
            
            for room in rooms:
                if room.area < min_area:
                    undersized_rooms.append(f"room_{room.id}")
            
            if undersized_rooms:
                return ConstraintViolation(
                    constraint_type=ConstraintType.ROOM_SIZE,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Undersized rooms: {', '.join(undersized_rooms)}",
                    affected_elements=undersized_rooms,
                    suggested_fix="Increase room size or combine with adjacent rooms",
                    confidence=0.8,
                    rule_id="RS001"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking room size: {str(e)}")
            return None
    
    def _check_room_proportions(self, graph_data, segmentation_result: Dict[str, Any], 
                              neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check room proportion constraints"""
        try:
            rooms = segmentation_result.get('rooms', [])
            disproportionate_rooms = []
            
            for room in rooms:
                if room.bounding_box:
                    x, y, w, h = room.bounding_box
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # Check if room is too narrow or too wide
                    if aspect_ratio > 3 or aspect_ratio < 0.33:
                        disproportionate_rooms.append(f"room_{room.id}")
            
            if disproportionate_rooms:
                return ConstraintViolation(
                    constraint_type=ConstraintType.ROOM_SIZE,
                    severity=ViolationSeverity.LOW,
                    description=f"Disproportionate rooms: {', '.join(disproportionate_rooms)}",
                    affected_elements=disproportionate_rooms,
                    suggested_fix="Adjust room proportions for better functionality",
                    confidence=0.5,
                    rule_id="RS002"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking room proportions: {str(e)}")
            return None
    
    def _check_accessibility(self, graph_data, segmentation_result: Dict[str, Any], 
                           neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check accessibility constraints"""
        try:
            # This is a simplified check - in practice, you'd implement pathfinding
            adjacency = segmentation_result.get('adjacency', {})
            rooms = segmentation_result.get('rooms', [])
            
            # Find rooms that are not reachable from room 0 (assuming it's the entrance)
            if 0 in adjacency:
                reachable_rooms = self._find_reachable_rooms(adjacency, 0)
                all_room_ids = set(range(len(rooms)))
                unreachable_rooms = all_room_ids - reachable_rooms
                
                if unreachable_rooms:
                    unreachable_list = [f"room_{rid}" for rid in unreachable_rooms]
                    return ConstraintViolation(
                        constraint_type=ConstraintType.ACCESSIBILITY,
                        severity=ViolationSeverity.HIGH,
                        description=f"Inaccessible rooms: {', '.join(unreachable_list)}",
                        affected_elements=unreachable_list,
                        suggested_fix="Add connecting doors or corridors",
                        confidence=0.9,
                        rule_id="AC001"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking accessibility: {str(e)}")
            return None
    
    def _check_wheelchair_access(self, graph_data, segmentation_result: Dict[str, Any], 
                               neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check wheelchair accessibility"""
        try:
            doors = segmentation_result.get('doors', [])
            narrow_doors = []
            
            min_width = 36  # Minimum width for wheelchair access in pixels
            
            for door in doors:
                if door.width < min_width:
                    narrow_doors.append(f"door_{door.id}")
            
            if narrow_doors:
                return ConstraintViolation(
                    constraint_type=ConstraintType.ACCESSIBILITY,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Doors too narrow for wheelchair access: {', '.join(narrow_doors)}",
                    affected_elements=narrow_doors,
                    suggested_fix="Increase door width to at least 36 inches",
                    confidence=0.8,
                    rule_id="AC002"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking wheelchair access: {str(e)}")
            return None
    
    def _check_structural_integrity(self, graph_data, segmentation_result: Dict[str, Any], 
                                  neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check structural integrity constraints"""
        try:
            # This is a placeholder - in practice, you'd implement structural analysis
            walls = segmentation_result.get('walls', [])
            
            # Check for very long walls without support
            long_walls = []
            max_length = 200  # Maximum wall length without support
            
            for wall in walls:
                if wall.length > max_length:
                    long_walls.append(f"wall_{wall.id}")
            
            if long_walls:
                return ConstraintViolation(
                    constraint_type=ConstraintType.STRUCTURAL_INTEGRITY,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Long walls without support: {', '.join(long_walls)}",
                    affected_elements=long_walls,
                    suggested_fix="Add structural support or break up long walls",
                    confidence=0.7,
                    rule_id="SI001"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking structural integrity: {str(e)}")
            return None
    
    def _check_circulation(self, graph_data, segmentation_result: Dict[str, Any], 
                         neural_scores: Dict[str, float]) -> Optional[ConstraintViolation]:
        """Check circulation efficiency"""
        try:
            # This is a simplified check - in practice, you'd implement circulation analysis
            adjacency = segmentation_result.get('adjacency', {})
            
            # Check for rooms with too many connections (inefficient)
            overconnected_rooms = []
            max_connections = 4
            
            for room_id, connections in adjacency.items():
                if len(connections) > max_connections:
                    overconnected_rooms.append(f"room_{room_id}")
            
            if overconnected_rooms:
                return ConstraintViolation(
                    constraint_type=ConstraintType.CIRCULATION,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Overconnected rooms: {', '.join(overconnected_rooms)}",
                    affected_elements=overconnected_rooms,
                    suggested_fix="Simplify circulation by reducing connections",
                    confidence=0.6,
                    rule_id="CI001"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking circulation: {str(e)}")
            return None
    
    def _point_to_line_distance(self, px: float, py: float, x1: float, y1: float, 
                               x2: float, y2: float) -> float:
        """Calculate distance from point to line"""
        try:
            # Line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            # Distance formula
            distance = abs(a * px + b * py + c) / np.sqrt(a * a + b * b)
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating point-to-line distance: {str(e)}")
            return float('inf')
    
    def _find_reachable_rooms(self, adjacency: Dict[int, List[int]], start_room: int) -> Set[int]:
        """Find all rooms reachable from a starting room"""
        try:
            visited = set()
            queue = [start_room]
            
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    if current in adjacency:
                        queue.extend(adjacency[current])
            
            return visited
            
        except Exception as e:
            logger.error(f"Error finding reachable rooms: {str(e)}")
            return {start_room}
    
    def _calculate_validity_score(self, violations: List[ConstraintViolation], 
                                rule_traces: List[Dict[str, Any]]) -> float:
        """Calculate overall validity score"""
        try:
            if not violations:
                return 1.0
            
            # Weight violations by severity and rule weight
            total_penalty = 0.0
            total_weight = 0.0
            
            for violation in violations:
                severity_penalty = {
                    ViolationSeverity.LOW: 0.1,
                    ViolationSeverity.MEDIUM: 0.3,
                    ViolationSeverity.HIGH: 0.6,
                    ViolationSeverity.CRITICAL: 1.0
                }
                
                # Find corresponding rule
                rule_weight = 1.0
                for rule_trace in rule_traces:
                    if rule_trace['rule_id'] == violation.rule_id:
                        rule_weight = rule_trace['weight']
                        break
                
                penalty = severity_penalty[violation.severity] * rule_weight * violation.confidence
                total_penalty += penalty
                total_weight += rule_weight
            
            # Calculate score (0-1, where 1 is perfect)
            if total_weight > 0:
                score = max(0.0, 1.0 - (total_penalty / total_weight))
            else:
                score = 1.0
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating validity score: {str(e)}")
            return 0.5
    
    def _generate_repair_suggestions(self, violations: List[ConstraintViolation]) -> List[Dict[str, Any]]:
        """Generate repair suggestions based on violations"""
        try:
            suggestions = []
            
            # Group violations by type
            violation_groups = {}
            for violation in violations:
                violation_type = violation.constraint_type.value
                if violation_type not in violation_groups:
                    violation_groups[violation_type] = []
                violation_groups[violation_type].append(violation)
            
            # Generate suggestions for each group
            for violation_type, group_violations in violation_groups.items():
                suggestion = {
                    'type': violation_type,
                    'priority': self._get_priority(group_violations),
                    'description': self._get_group_description(violation_type, group_violations),
                    'steps': self._get_repair_steps(violation_type, group_violations),
                    'affected_elements': [elem for v in group_violations for elem in v.affected_elements]
                }
                suggestions.append(suggestion)
            
            # Sort by priority
            suggestions.sort(key=lambda x: x['priority'], reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating repair suggestions: {str(e)}")
            return []
    
    def _get_priority(self, violations: List[ConstraintViolation]) -> float:
        """Get priority score for a group of violations"""
        try:
            severity_scores = {
                ViolationSeverity.LOW: 1,
                ViolationSeverity.MEDIUM: 2,
                ViolationSeverity.HIGH: 3,
                ViolationSeverity.CRITICAL: 4
            }
            
            max_severity = max(severity_scores[v.severity] for v in violations)
            avg_confidence = np.mean([v.confidence for v in violations])
            
            return max_severity * avg_confidence
            
        except Exception as e:
            logger.error(f"Error getting priority: {str(e)}")
            return 1.0
    
    def _get_group_description(self, violation_type: str, violations: List[ConstraintViolation]) -> str:
        """Get description for a group of violations"""
        descriptions = {
            'room_connectivity': f"Found {len(violations)} room connectivity issues",
            'door_placement': f"Found {len(violations)} door placement issues",
            'room_size': f"Found {len(violations)} room size issues",
            'accessibility': f"Found {len(violations)} accessibility issues",
            'structural_integrity': f"Found {len(violations)} structural issues",
            'circulation': f"Found {len(violations)} circulation issues"
        }
        return descriptions.get(violation_type, f"Found {len(violations)} issues")
    
    def _get_repair_steps(self, violation_type: str, violations: List[ConstraintViolation]) -> List[str]:
        """Get repair steps for a violation type"""
        steps_map = {
            'room_connectivity': [
                "Identify isolated or poorly connected rooms",
                "Add doors between disconnected rooms",
                "Create corridors for better circulation",
                "Verify all rooms are accessible"
            ],
            'door_placement': [
                "Review door positions relative to walls",
                "Ensure doors are properly aligned with openings",
                "Check door width meets accessibility standards",
                "Verify door swing clearance"
            ],
            'room_size': [
                "Measure room dimensions",
                "Compare against minimum size requirements",
                "Consider combining undersized rooms",
                "Adjust room proportions if needed"
            ],
            'accessibility': [
                "Check accessibility from main entrance",
                "Verify wheelchair access requirements",
                "Ensure proper door widths",
                "Review circulation paths"
            ],
            'structural_integrity': [
                "Identify load-bearing elements",
                "Check wall lengths and support requirements",
                "Verify structural connections",
                "Consult structural engineer if needed"
            ],
            'circulation': [
                "Analyze traffic flow patterns",
                "Simplify overconnected areas",
                "Optimize circulation efficiency",
                "Ensure logical room adjacencies"
            ]
        }
        return steps_map.get(violation_type, ["Review and address identified issues"])
    
    def _create_summary(self, violations: List[ConstraintViolation], validity_score: float) -> Dict[str, Any]:
        """Create summary of constraint analysis"""
        try:
            severity_counts = {
                'low': 0,
                'medium': 0,
                'high': 0,
                'critical': 0
            }
            
            for violation in violations:
                severity_counts[violation.severity.value] += 1
            
            return {
                'total_violations': len(violations),
                'severity_breakdown': severity_counts,
                'validity_score': validity_score,
                'overall_status': 'good' if validity_score > 0.8 else 'needs_improvement' if validity_score > 0.5 else 'poor',
                'critical_issues': severity_counts['critical'],
                'high_priority_issues': severity_counts['high'] + severity_counts['critical']
            }
            
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return {}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_violation_history(self) -> List[Dict[str, Any]]:
        """Get violation history"""
        return self.violation_history
    
    def clear_history(self):
        """Clear violation history"""
        self.violation_history = []
    
    def export_constraints(self, filepath: str):
        """Export constraint rules to file"""
        try:
            rules_data = []
            for rule in self.constraint_rules:
                rules_data.append({
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'constraint_type': rule.constraint_type.value,
                    'severity': rule.severity.value,
                    'weight': rule.weight
                })
            
            with open(filepath, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            logger.info(f"Constraints exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting constraints: {str(e)}")
