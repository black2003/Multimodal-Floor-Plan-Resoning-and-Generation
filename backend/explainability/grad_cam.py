"""
Grad-CAM implementation for image explainability
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

class GradCAMExplainer:
    """Grad-CAM explainer for vision transformer attention visualization"""
    
    def __init__(self, model, target_layer: str = "blocks.11.norm1"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
    def register_hooks(self):
        """Register forward and backward hooks"""
        try:
            # Get the target layer
            target_module = self._get_target_layer()
            
            if target_module is None:
                logger.warning(f"Target layer {self.target_layer} not found")
                return
            
            # Forward hook to capture activations
            def forward_hook(module, input, output):
                self.activations = output.detach()
            
            # Backward hook to capture gradients
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()
            
            # Register hooks
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_backward_hook(backward_hook))
            
        except Exception as e:
            logger.error(f"Error registering hooks: {str(e)}")
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _get_target_layer(self):
        """Get the target layer from the model"""
        try:
            # Navigate to the target layer
            parts = self.target_layer.split('.')
            module = self.model
            
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                elif part.isdigit():
                    module = module[int(part)]
                else:
                    return None
            
            return module
            
        except Exception as e:
            logger.error(f"Error getting target layer: {str(e)}")
            return None
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM for the input
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index (if None, uses highest scoring class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        try:
            # Register hooks
            self.register_hooks()
            
            # Forward pass
            self.model.eval()
            output = self.model(input_tensor)
            
            # Get target class
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            target = output[0, class_idx]
            target.backward(retain_graph=True)
            
            # Generate CAM
            cam = self._compute_cam()
            
            # Remove hooks
            self.remove_hooks()
            
            return cam
            
        except Exception as e:
            logger.error(f"Error generating CAM: {str(e)}")
            self.remove_hooks()
            return np.zeros((224, 224))
    
    def _compute_cam(self) -> np.ndarray:
        """Compute the Grad-CAM heatmap"""
        try:
            if self.gradients is None or self.activations is None:
                return np.zeros((224, 224))
            
            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Resize to input size
            cam = cv2.resize(cam, (224, 224))
            
            return cam
            
        except Exception as e:
            logger.error(f"Error computing CAM: {str(e)}")
            return np.zeros((224, 224))
    
    def visualize_cam(self, original_image: np.ndarray, cam: np.ndarray, 
                     alpha: float = 0.4) -> np.ndarray:
        """
        Visualize Grad-CAM overlay on original image
        
        Args:
            original_image: Original image as numpy array
            cam: Grad-CAM heatmap
            alpha: Overlay transparency
            
        Returns:
            Visualization image
        """
        try:
            # Resize original image to match CAM
            if original_image.shape[:2] != cam.shape:
                original_image = cv2.resize(original_image, (cam.shape[1], cam.shape[0]))
            
            # Convert CAM to heatmap
            heatmap = cm.jet(cam)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # Convert original image to RGB if needed
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                original_rgb = original_image
            else:
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            
            # Blend images
            overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error visualizing CAM: {str(e)}")
            return original_image
    
    def get_attention_regions(self, cam: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract attention regions from CAM
        
        Args:
            cam: Grad-CAM heatmap
            threshold: Threshold for attention regions
            
        Returns:
            List of attention regions
        """
        try:
            # Create binary mask
            binary_mask = (cam > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for i, contour in enumerate(contours):
                # Calculate area
                area = cv2.contourArea(contour)
                
                if area < 100:  # Filter small regions
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate attention score
                roi = cam[y:y+h, x:x+w]
                attention_score = np.mean(roi)
                
                regions.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'attention_score': attention_score,
                    'contour': contour
                })
            
            # Sort by attention score
            regions.sort(key=lambda x: x['attention_score'], reverse=True)
            
            return regions
            
        except Exception as e:
            logger.error(f"Error extracting attention regions: {str(e)}")
            return []
    
    def explain_prediction(self, input_tensor: torch.Tensor, 
                          original_image: np.ndarray) -> Dict[str, Any]:
        """
        Generate complete explanation for prediction
        
        Args:
            input_tensor: Input image tensor
            original_image: Original image as numpy array
            
        Returns:
            Dictionary containing explanation results
        """
        try:
            # Generate CAM
            cam = self.generate_cam(input_tensor)
            
            # Create visualization
            visualization = self.visualize_cam(original_image, cam)
            
            # Extract attention regions
            regions = self.get_attention_regions(cam)
            
            # Get top attention areas
            top_regions = regions[:5]  # Top 5 regions
            
            return {
                'cam_heatmap': cam,
                'visualization': visualization,
                'attention_regions': regions,
                'top_regions': top_regions,
                'max_attention': float(np.max(cam)),
                'mean_attention': float(np.mean(cam)),
                'attention_distribution': {
                    'min': float(np.min(cam)),
                    'max': float(np.max(cam)),
                    'mean': float(np.mean(cam)),
                    'std': float(np.std(cam))
                }
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {}
    
    def save_explanation(self, explanation: Dict[str, Any], save_path: str):
        """Save explanation results to files"""
        try:
            # Save CAM heatmap
            if 'cam_heatmap' in explanation:
                plt.figure(figsize=(8, 8))
                plt.imshow(explanation['cam_heatmap'], cmap='jet')
                plt.colorbar()
                plt.title('Grad-CAM Heatmap')
                plt.axis('off')
                plt.savefig(f"{save_path}_cam.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save visualization
            if 'visualization' in explanation:
                cv2.imwrite(f"{save_path}_overlay.png", 
                           cv2.cvtColor(explanation['visualization'], cv2.COLOR_RGB2BGR))
            
            # Save metadata
            metadata = {
                'max_attention': explanation.get('max_attention', 0),
                'mean_attention': explanation.get('mean_attention', 0),
                'num_regions': len(explanation.get('attention_regions', [])),
                'top_regions': explanation.get('top_regions', [])
            }
            
            with open(f"{save_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Explanation saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving explanation: {str(e)}")
    
    def compare_explanations(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple explanations"""
        try:
            if not explanations:
                return {}
            
            # Extract attention scores
            attention_scores = [exp.get('mean_attention', 0) for exp in explanations]
            max_attentions = [exp.get('max_attention', 0) for exp in explanations]
            
            # Calculate statistics
            comparison = {
                'mean_attention_scores': attention_scores,
                'max_attention_scores': max_attentions,
                'avg_mean_attention': np.mean(attention_scores),
                'avg_max_attention': np.mean(max_attentions),
                'attention_variance': np.var(attention_scores),
                'num_explanations': len(explanations)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing explanations: {str(e)}")
            return {}
