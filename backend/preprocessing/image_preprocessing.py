"""
Image preprocessing module for floor plan images
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing for floor plan analysis"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define transforms for different purposes
        self.normalize_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            if augment:
                tensor = self.augment_transform(image)
            else:
                tensor = self.normalize_transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def enhance_floorplan(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance floor plan image for better segmentation
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error enhancing floor plan: {str(e)}")
            return image
    
    def detect_orientation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect the orientation of the floor plan
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with orientation information
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Analyze line orientations
                horizontal_lines = 0
                vertical_lines = 0
                
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    
                    # Check if line is horizontal or vertical
                    if abs(angle) < 10 or abs(angle - 180) < 10:
                        horizontal_lines += 1
                    elif abs(angle - 90) < 10:
                        vertical_lines += 1
                
                # Determine dominant orientation
                if horizontal_lines > vertical_lines:
                    orientation = 'landscape'
                else:
                    orientation = 'portrait'
            else:
                orientation = 'unknown'
            
            return {
                'orientation': orientation,
                'horizontal_lines': horizontal_lines,
                'vertical_lines': vertical_lines,
                'total_lines': len(lines) if lines is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting orientation: {str(e)}")
            return {'orientation': 'unknown', 'error': str(e)}
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract basic features from floor plan image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with extracted features
        """
        try:
            features = {}
            
            # Basic image properties
            features['height'] = image.shape[0]
            features['width'] = image.shape[1]
            features['channels'] = image.shape[2] if len(image.shape) == 3 else 1
            
            # Color statistics
            if len(image.shape) == 3:
                features['mean_color'] = np.mean(image, axis=(0, 1)).tolist()
                features['std_color'] = np.std(image, axis=(0, 1)).tolist()
            else:
                features['mean_intensity'] = float(np.mean(image))
                features['std_intensity'] = float(np.std(image))
            
            # Edge density
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            # Orientation information
            orientation_info = self.detect_orientation(image)
            features.update(orientation_info)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {'error': str(e)}
    
    def resize_with_aspect_ratio(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        try:
            # Calculate aspect ratio
            original_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            
            if original_ratio > target_ratio:
                # Image is wider, fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / original_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * original_ratio)
            
            # Resize image
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            result = Image.new('RGB', target_size, (255, 255, 255))
            result.paste(resized, ((target_size[0] - new_width) // 2, 
                                 (target_size[1] - new_height) // 2))
            
            return result
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image
