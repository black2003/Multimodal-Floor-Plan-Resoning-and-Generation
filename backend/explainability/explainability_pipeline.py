"""
Complete explainability pipeline for multi-modal floor plan understanding
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import base64
import io

from .grad_cam import GradCAMExplainer
from .attention_visualizer import AttentionVisualizer
from .constraint_tracer import ConstraintTracer

logger = logging.getLogger(__name__)

class ExplainabilityPipeline:
    """Complete explainability pipeline for multi-modal analysis"""
    
    def __init__(self, model=None):
        self.model = model
        self.grad_cam_explainer = GradCAMExplainer(model) if model else None
        self.attention_visualizer = AttentionVisualizer()
        self.constraint_tracer = ConstraintTracer()
        
    def explain_analysis(self, image: Image.Image, analysis_result: Dict[str, Any], 
                        modality: str = 'all') -> Dict[str, Any]:
        """
        Generate comprehensive explanations for analysis results
        
        Args:
            image: Original floor plan image
            analysis_result: Results from the main analysis pipeline
            modality: Which modalities to explain ('all', 'image', 'text', 'graph', 'constraints')
            
        Returns:
            Dictionary containing all explanations
        """
        try:
            explanations = {}
            
            # Image explanations
            if modality in ['all', 'image']:
                explanations['image'] = self._explain_image_modality(image, analysis_result)
            
            # Text explanations
            if modality in ['all', 'text']:
                explanations['text'] = self._explain_text_modality(analysis_result)
            
            # Graph explanations
            if modality in ['all', 'graph']:
                explanations['graph'] = self._explain_graph_modality(analysis_result)
            
            # Constraint explanations
            if modality in ['all', 'constraints']:
                explanations['constraints'] = self._explain_constraints(analysis_result)
            
            # Cross-modal explanations
            if modality == 'all':
                explanations['cross_modal'] = self._explain_cross_modal_interactions(analysis_result)
            
            # Summary
            explanations['summary'] = self._create_explanation_summary(explanations)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            return {}
    
    def _explain_image_modality(self, image: Image.Image, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image modality explanations"""
        try:
            explanations = {}
            
            # Convert image to numpy array
            image_array = np.array(image)
            
            # Grad-CAM explanation (if model is available)
            if self.grad_cam_explainer and self.model:
                try:
                    # This would require the original input tensor
                    # For now, create a placeholder
                    explanations['grad_cam'] = {
                        'heatmap': 'placeholder_heatmap',
                        'important_regions': 'placeholder_regions',
                        'attention_weights': 'placeholder_attention'
                    }
                except Exception as e:
                    logger.warning(f"Grad-CAM explanation failed: {str(e)}")
                    explanations['grad_cam'] = {'error': str(e)}
            
            # Image attention visualization
            if 'attention_weights' in analysis_result.get('metadata', {}):
                attention_weights = analysis_result['metadata']['attention_weights']
                if 'image_attention' in attention_weights:
                    img_attn_viz = self.attention_visualizer.visualize_image_attention(
                        attention_weights['image_attention']
                    )
                    explanations['attention_visualization'] = img_attn_viz
            
            # Image features analysis
            explanations['image_features'] = self._analyze_image_features(image_array)
            
            # Segmentation analysis
            if 'segmentation_info' in analysis_result.get('metadata', {}):
                seg_info = analysis_result['metadata']['segmentation_info']
                explanations['segmentation_analysis'] = {
                    'num_rooms': seg_info.get('num_rooms', 0),
                    'num_doors': seg_info.get('num_doors', 0),
                    'num_walls': seg_info.get('num_walls', 0),
                    'complexity_score': self._calculate_complexity_score(seg_info)
                }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining image modality: {str(e)}")
            return {'error': str(e)}
    
    def _explain_text_modality(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text modality explanations"""
        try:
            explanations = {}
            
            # VQA explanations
            if 'vqa' in analysis_result:
                vqa_result = analysis_result['vqa']
                explanations['vqa_analysis'] = {
                    'answer': vqa_result.get('answer', ''),
                    'confidence': vqa_result.get('confidence', 0.0),
                    'answer_type': vqa_result.get('answer_type', 0),
                    'reasoning': self._generate_vqa_reasoning(vqa_result)
                }
            
            # Text attention visualization
            if 'attention_weights' in analysis_result.get('metadata', {}):
                attention_weights = analysis_result['metadata']['attention_weights']
                if 'text_attention' in attention_weights:
                    # This would require token information
                    explanations['text_attention'] = {
                        'important_tokens': 'placeholder_tokens',
                        'attention_weights': 'placeholder_weights'
                    }
            
            # Query analysis
            if 'text_query' in analysis_result.get('metadata', {}):
                query = analysis_result['metadata']['text_query']
                explanations['query_analysis'] = {
                    'query_type': self._classify_query_type(query),
                    'complexity': self._assess_query_complexity(query),
                    'keywords': self._extract_keywords(query)
                }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining text modality: {str(e)}")
            return {'error': str(e)}
    
    def _explain_graph_modality(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate graph modality explanations"""
        try:
            explanations = {}
            
            # Graph structure analysis
            if 'segmentation_info' in analysis_result.get('metadata', {}):
                seg_info = analysis_result['metadata']['segmentation_info']
                explanations['graph_structure'] = {
                    'num_nodes': seg_info.get('num_rooms', 0) + seg_info.get('num_doors', 0),
                    'num_edges': seg_info.get('num_doors', 0) * 2,  # Approximate
                    'connectivity_score': self._calculate_connectivity_score(seg_info),
                    'graph_density': self._calculate_graph_density(seg_info)
                }
            
            # Graph attention visualization
            if 'attention_weights' in analysis_result.get('metadata', {}):
                attention_weights = analysis_result['metadata']['attention_weights']
                if 'graph_attention' in attention_weights:
                    explanations['graph_attention'] = {
                        'important_nodes': 'placeholder_nodes',
                        'attention_weights': 'placeholder_weights'
                    }
            
            # Room relationship analysis
            explanations['room_relationships'] = self._analyze_room_relationships(analysis_result)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining graph modality: {str(e)}")
            return {'error': str(e)}
    
    def _explain_constraints(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate constraint explanations"""
        try:
            explanations = {}
            
            # Validity analysis
            if 'validity' in analysis_result:
                validity_result = analysis_result['validity']
                explanations['validity_analysis'] = {
                    'validity_score': validity_result.get('validity_score', 0.0),
                    'issues': validity_result.get('issues', {}),
                    'suggestions': validity_result.get('suggestions', {}),
                    'constraint_scores': validity_result.get('constraint_scores', {})
                }
            
            # Constraint violations
            if 'constraint_violations' in analysis_result:
                violations = analysis_result['constraint_violations']
                explanations['violations'] = {
                    'total_violations': len(violations),
                    'severity_breakdown': self._categorize_violations(violations),
                    'critical_issues': [v for v in violations if v.get('severity') == 'critical'],
                    'repair_priority': self._prioritize_repairs(violations)
                }
            
            # Rule trace
            explanations['rule_trace'] = self._generate_rule_trace(analysis_result)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining constraints: {str(e)}")
            return {'error': str(e)}
    
    def _explain_cross_modal_interactions(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-modal interaction explanations"""
        try:
            explanations = {}
            
            # Modality importance
            explanations['modality_importance'] = self._assess_modality_importance(analysis_result)
            
            # Cross-modal attention
            if 'attention_weights' in analysis_result.get('metadata', {}):
                attention_weights = analysis_result['metadata']['attention_weights']
                if 'cross_modal_attention' in attention_weights:
                    cross_modal_viz = self.attention_visualizer.visualize_cross_modal_attention(
                        attention_weights['cross_modal_attention']
                    )
                    explanations['cross_modal_attention'] = cross_modal_viz
            
            # Interaction patterns
            explanations['interaction_patterns'] = self._analyze_interaction_patterns(analysis_result)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining cross-modal interactions: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_image_features(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze basic image features"""
        try:
            # Convert to grayscale for analysis
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
            
            # Basic statistics
            features = {
                'image_size': image_array.shape[:2],
                'mean_intensity': float(np.mean(gray)),
                'std_intensity': float(np.std(gray)),
                'edge_density': self._calculate_edge_density(gray),
                'texture_complexity': self._calculate_texture_complexity(gray),
                'symmetry_score': self._calculate_symmetry_score(gray)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing image features: {str(e)}")
            return {}
    
    def _calculate_complexity_score(self, seg_info: Dict[str, Any]) -> float:
        """Calculate floor plan complexity score"""
        try:
            num_rooms = seg_info.get('num_rooms', 0)
            num_doors = seg_info.get('num_doors', 0)
            num_walls = seg_info.get('num_walls', 0)
            
            # Simple complexity metric
            complexity = (num_rooms * 0.3 + num_doors * 0.2 + num_walls * 0.1) / 10.0
            return min(1.0, complexity)
            
        except Exception as e:
            logger.error(f"Error calculating complexity score: {str(e)}")
            return 0.5
    
    def _generate_vqa_reasoning(self, vqa_result: Dict[str, Any]) -> str:
        """Generate reasoning for VQA answer"""
        try:
            answer = vqa_result.get('answer', '')
            confidence = vqa_result.get('confidence', 0.0)
            answer_type = vqa_result.get('answer_type', 0)
            
            reasoning = f"Based on the floor plan analysis, the answer is '{answer}' "
            reasoning += f"with {confidence:.2f} confidence. "
            
            if answer_type == 0:  # number
                reasoning += "This is a numerical answer based on counting elements in the floor plan."
            elif answer_type == 1:  # yes_no
                reasoning += "This is a yes/no answer based on the presence or absence of specific features."
            elif answer_type == 2:  # color
                reasoning += "This is a color answer based on visual analysis of the floor plan."
            elif answer_type == 3:  # location
                reasoning += "This is a location answer based on spatial relationships in the floor plan."
            else:  # text
                reasoning += "This is a descriptive answer based on the overall floor plan structure."
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating VQA reasoning: {str(e)}")
            return "Unable to generate reasoning for this answer."
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'which', 'where']):
            return 'descriptive'
        elif any(word in query_lower for word in ['how many', 'count', 'number']):
            return 'counting'
        elif any(word in query_lower for word in ['is', 'are', 'does', 'do']):
            return 'yes_no'
        elif any(word in query_lower for word in ['size', 'area', 'dimension']):
            return 'measurement'
        else:
            return 'general'
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity"""
        words = query.split()
        
        if len(words) <= 3:
            return 'simple'
        elif len(words) <= 8:
            return 'medium'
        else:
            return 'complex'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords[:5]  # Top 5 keywords
    
    def _calculate_connectivity_score(self, seg_info: Dict[str, Any]) -> float:
        """Calculate connectivity score"""
        try:
            num_rooms = seg_info.get('num_rooms', 0)
            num_doors = seg_info.get('num_doors', 0)
            
            if num_rooms == 0:
                return 0.0
            
            # Connectivity ratio
            connectivity = num_doors / num_rooms
            return min(1.0, connectivity / 2.0)  # Normalize
            
        except Exception as e:
            logger.error(f"Error calculating connectivity score: {str(e)}")
            return 0.5
    
    def _calculate_graph_density(self, seg_info: Dict[str, Any]) -> float:
        """Calculate graph density"""
        try:
            num_rooms = seg_info.get('num_rooms', 0)
            num_doors = seg_info.get('num_doors', 0)
            
            if num_rooms <= 1:
                return 0.0
            
            # Maximum possible edges (complete graph)
            max_edges = num_rooms * (num_rooms - 1) / 2
            
            # Actual edges (approximate)
            actual_edges = num_doors * 2  # Each door connects 2 rooms
            
            density = actual_edges / max_edges
            return min(1.0, density)
            
        except Exception as e:
            logger.error(f"Error calculating graph density: {str(e)}")
            return 0.5
    
    def _analyze_room_relationships(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze room relationships"""
        try:
            # This would analyze the actual room relationships from the graph
            # For now, return placeholder analysis
            return {
                'adjacent_rooms': 'placeholder_adjacent',
                'room_hierarchy': 'placeholder_hierarchy',
                'circulation_patterns': 'placeholder_circulation'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing room relationships: {str(e)}")
            return {}
    
    def _categorize_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize violations by severity"""
        try:
            categories = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            
            for violation in violations:
                severity = violation.get('severity', 'low')
                if severity in categories:
                    categories[severity] += 1
            
            return categories
            
        except Exception as e:
            logger.error(f"Error categorizing violations: {str(e)}")
            return {}
    
    def _prioritize_repairs(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize repair suggestions"""
        try:
            # Sort violations by severity and confidence
            priority_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            
            prioritized = sorted(violations, key=lambda v: (
                priority_map.get(v.get('severity', 'low'), 1),
                v.get('confidence', 0.0)
            ), reverse=True)
            
            return prioritized[:5]  # Top 5 priorities
            
        except Exception as e:
            logger.error(f"Error prioritizing repairs: {str(e)}")
            return []
    
    def _generate_rule_trace(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate rule trace for constraint checking"""
        try:
            # This would trace the actual rule execution
            # For now, return placeholder trace
            return [
                {
                    'rule_id': 'RC001',
                    'rule_name': 'No Isolated Rooms',
                    'status': 'passed',
                    'execution_time': '0.001s'
                },
                {
                    'rule_id': 'DP001',
                    'rule_name': 'Proper Door Placement',
                    'status': 'failed',
                    'execution_time': '0.002s'
                }
            ]
            
        except Exception as e:
            logger.error(f"Error generating rule trace: {str(e)}")
            return []
    
    def _assess_modality_importance(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Assess importance of each modality"""
        try:
            # This would analyze the actual modality contributions
            # For now, return placeholder importance scores
            return {
                'image': 0.4,
                'text': 0.3,
                'graph': 0.3
            }
            
        except Exception as e:
            logger.error(f"Error assessing modality importance: {str(e)}")
            return {}
    
    def _analyze_interaction_patterns(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-modal interaction patterns"""
        try:
            # This would analyze actual interaction patterns
            # For now, return placeholder patterns
            return {
                'image_text_correlation': 0.7,
                'text_graph_correlation': 0.6,
                'image_graph_correlation': 0.8,
                'dominant_interaction': 'image_graph'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {str(e)}")
            return {}
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density in image"""
        try:
            import cv2
            edges = cv2.Canny(gray_image, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            return edge_pixels / total_pixels
            
        except Exception as e:
            logger.error(f"Error calculating edge density: {str(e)}")
            return 0.0
    
    def _calculate_texture_complexity(self, gray_image: np.ndarray) -> float:
        """Calculate texture complexity"""
        try:
            # Simple texture complexity using standard deviation
            return float(np.std(gray_image)) / 255.0
            
        except Exception as e:
            logger.error(f"Error calculating texture complexity: {str(e)}")
            return 0.0
    
    def _calculate_symmetry_score(self, gray_image: np.ndarray) -> float:
        """Calculate symmetry score"""
        try:
            # Simple horizontal symmetry check
            h, w = gray_image.shape
            left_half = gray_image[:, :w//2]
            right_half = np.fliplr(gray_image[:, w//2:])
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate similarity
            similarity = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            return max(0.0, similarity) if not np.isnan(similarity) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating symmetry score: {str(e)}")
            return 0.0
    
    def _create_explanation_summary(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of all explanations"""
        try:
            summary = {
                'total_explanations': len(explanations),
                'modalities_explained': list(explanations.keys()),
                'explanation_quality': self._assess_explanation_quality(explanations),
                'key_insights': self._extract_key_insights(explanations),
                'recommendations': self._generate_recommendations(explanations)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating explanation summary: {str(e)}")
            return {}
    
    def _assess_explanation_quality(self, explanations: Dict[str, Any]) -> str:
        """Assess overall explanation quality"""
        try:
            quality_indicators = []
            
            for modality, explanation in explanations.items():
                if 'error' not in explanation:
                    quality_indicators.append(1)
                else:
                    quality_indicators.append(0)
            
            quality_score = np.mean(quality_indicators)
            
            if quality_score >= 0.8:
                return 'high'
            elif quality_score >= 0.6:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing explanation quality: {str(e)}")
            return 'unknown'
    
    def _extract_key_insights(self, explanations: Dict[str, Any]) -> List[str]:
        """Extract key insights from explanations"""
        try:
            insights = []
            
            # Extract insights from each modality
            for modality, explanation in explanations.items():
                if modality == 'image' and 'segmentation_analysis' in explanation:
                    seg_analysis = explanation['segmentation_analysis']
                    insights.append(f"Floor plan has {seg_analysis.get('num_rooms', 0)} rooms with complexity score {seg_analysis.get('complexity_score', 0):.2f}")
                
                elif modality == 'text' and 'vqa_analysis' in explanation:
                    vqa_analysis = explanation['vqa_analysis']
                    insights.append(f"VQA answer: '{vqa_analysis.get('answer', '')}' with confidence {vqa_analysis.get('confidence', 0):.2f}")
                
                elif modality == 'constraints' and 'validity_analysis' in explanation:
                    validity_analysis = explanation['validity_analysis']
                    insights.append(f"Floor plan validity score: {validity_analysis.get('validity_score', 0):.2f}")
            
            return insights[:5]  # Top 5 insights
            
        except Exception as e:
            logger.error(f"Error extracting key insights: {str(e)}")
            return []
    
    def _generate_recommendations(self, explanations: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on explanations"""
        try:
            recommendations = []
            
            # Generate recommendations based on findings
            if 'constraints' in explanations and 'validity_analysis' in explanations['constraints']:
                validity_score = explanations['constraints']['validity_analysis'].get('validity_score', 1.0)
                if validity_score < 0.7:
                    recommendations.append("Consider addressing structural and connectivity issues to improve floor plan validity")
            
            if 'image' in explanations and 'segmentation_analysis' in explanations['image']:
                complexity = explanations['image']['segmentation_analysis'].get('complexity_score', 0.5)
                if complexity > 0.8:
                    recommendations.append("Floor plan is quite complex - consider simplifying the layout for better usability")
            
            if 'graph' in explanations and 'graph_structure' in explanations['graph']:
                connectivity = explanations['graph']['graph_structure'].get('connectivity_score', 0.5)
                if connectivity < 0.3:
                    recommendations.append("Improve room connectivity by adding more doors or corridors")
            
            return recommendations[:3]  # Top 3 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def save_explanations(self, explanations: Dict[str, Any], save_path: str):
        """Save explanations to files"""
        try:
            # Save main explanations
            with open(f"{save_path}_explanations.json", 'w') as f:
                json.dump(explanations, f, indent=2, default=str)
            
            # Save visualizations if available
            for modality, explanation in explanations.items():
                if 'fig' in explanation:
                    explanation['fig'].savefig(f"{save_path}_{modality}_viz.png", dpi=300, bbox_inches='tight')
                    plt.close(explanation['fig'])
            
            logger.info(f"Explanations saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving explanations: {str(e)}")
    
    def export_explanation_report(self, explanations: Dict[str, Any], output_path: str):
        """Export comprehensive explanation report"""
        try:
            report = {
                'timestamp': self._get_timestamp(),
                'summary': explanations.get('summary', {}),
                'modality_explanations': {k: v for k, v in explanations.items() if k != 'summary'},
                'metadata': {
                    'total_modalities': len([k for k in explanations.keys() if k != 'summary']),
                    'explanation_quality': explanations.get('summary', {}).get('explanation_quality', 'unknown')
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Explanation report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting explanation report: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
