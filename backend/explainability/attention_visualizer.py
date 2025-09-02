"""
Attention visualization for multi-modal models
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """Visualizer for attention weights across modalities"""
    
    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
    def visualize_image_attention(self, attention_weights: torch.Tensor, 
                                image_size: Tuple[int, int] = (224, 224),
                                patch_size: int = 16) -> Dict[str, Any]:
        """
        Visualize attention weights for image patches
        
        Args:
            attention_weights: Attention weights [num_heads, num_patches, num_patches]
            image_size: Size of the original image
            patch_size: Size of each patch
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            num_heads, num_patches, _ = attention_weights.shape
            
            # Calculate number of patches per dimension
            patches_per_dim = int(np.sqrt(num_patches))
            
            # Average attention across heads
            avg_attention = torch.mean(attention_weights, dim=0)
            
            # Get attention for CLS token (first patch)
            cls_attention = avg_attention[0, 1:].reshape(patches_per_dim, patches_per_dim)
            
            # Create attention heatmap
            attention_map = cls_attention.cpu().numpy()
            
            # Resize to image size
            attention_map_resized = self._resize_attention_map(attention_map, image_size)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original attention map
            im1 = axes[0].imshow(attention_map, cmap='hot', interpolation='nearest')
            axes[0].set_title('Patch Attention Map')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            # Resized attention map
            im2 = axes[1].imshow(attention_map_resized, cmap='hot', interpolation='bilinear')
            axes[1].set_title('Resized Attention Map')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1])
            
            # Top attention patches
            top_patches = self._get_top_attention_patches(attention_map, top_k=10)
            axes[2].imshow(attention_map, cmap='hot', interpolation='nearest')
            self._highlight_patches(axes[2], top_patches, patches_per_dim)
            axes[2].set_title('Top Attention Patches')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            return {
                'attention_map': attention_map,
                'attention_map_resized': attention_map_resized,
                'top_patches': top_patches,
                'fig': fig,
                'statistics': {
                    'max_attention': float(np.max(attention_map)),
                    'mean_attention': float(np.mean(attention_map)),
                    'std_attention': float(np.std(attention_map))
                }
            }
            
        except Exception as e:
            logger.error(f"Error visualizing image attention: {str(e)}")
            return {}
    
    def visualize_text_attention(self, attention_weights: List[torch.Tensor], 
                               tokens: List[str]) -> Dict[str, Any]:
        """
        Visualize attention weights for text tokens
        
        Args:
            attention_weights: List of attention weights from different layers
            tokens: List of token strings
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            if not attention_weights or not tokens:
                return {}
            
            # Use the last layer's attention
            last_layer_attention = attention_weights[-1]
            
            # Average across heads
            if len(last_layer_attention.shape) == 4:  # [batch, heads, seq, seq]
                attention = torch.mean(last_layer_attention[0], dim=0)  # [seq, seq]
            else:  # [heads, seq, seq]
                attention = torch.mean(last_layer_attention, dim=0)
            
            # Get attention from CLS token to other tokens
            cls_attention = attention[0, 1:].cpu().numpy()  # Skip CLS token itself
            
            # Create token attention visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Bar plot of attention weights
            token_indices = range(len(tokens[1:]))  # Skip CLS token
            bars = ax1.bar(token_indices, cls_attention)
            ax1.set_xlabel('Token Index')
            ax1.set_ylabel('Attention Weight')
            ax1.set_title('Token Attention Weights')
            ax1.set_xticks(token_indices)
            ax1.set_xticklabels(tokens[1:], rotation=45, ha='right')
            
            # Color bars by attention weight
            colors = plt.cm.Reds(cls_attention / np.max(cls_attention))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Heatmap of full attention matrix
            im = ax2.imshow(attention.cpu().numpy(), cmap='Blues', aspect='auto')
            ax2.set_xlabel('Key Tokens')
            ax2.set_ylabel('Query Tokens')
            ax2.set_title('Full Attention Matrix')
            ax2.set_xticks(range(len(tokens)))
            ax2.set_xticklabels(tokens, rotation=45, ha='right')
            ax2.set_yticks(range(len(tokens)))
            ax2.set_yticklabels(tokens)
            plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            
            # Get top attended tokens
            top_token_indices = np.argsort(cls_attention)[-5:][::-1]
            top_tokens = [(tokens[i+1], cls_attention[i]) for i in top_token_indices]
            
            return {
                'attention_matrix': attention.cpu().numpy(),
                'token_attention': cls_attention,
                'top_tokens': top_tokens,
                'fig': fig,
                'statistics': {
                    'max_attention': float(np.max(cls_attention)),
                    'mean_attention': float(np.mean(cls_attention)),
                    'num_tokens': len(tokens)
                }
            }
            
        except Exception as e:
            logger.error(f"Error visualizing text attention: {str(e)}")
            return {}
    
    def visualize_graph_attention(self, attention_weights: List[torch.Tensor], 
                                graph_data, node_positions: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Visualize attention weights for graph nodes and edges
        
        Args:
            attention_weights: List of attention weights from GNN layers
            graph_data: PyTorch Geometric Data object
            node_positions: Optional node positions for layout
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            if not attention_weights:
                return {}
            
            # Use the last layer's attention
            last_layer_attention = attention_weights[-1]
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            num_nodes = graph_data.num_nodes
            for i in range(num_nodes):
                node_type = graph_data.node_types[i] if hasattr(graph_data, 'node_types') else 'unknown'
                G.add_node(i, node_type=node_type)
            
            # Add edges
            edge_index = graph_data.edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                source, target = edge_index[:, i]
                G.add_edge(int(source), int(target))
            
            # Calculate node attention (sum of incoming attention)
            node_attention = torch.zeros(num_nodes)
            for i in range(num_nodes):
                # Sum attention from all other nodes
                attention_sum = 0
                for j in range(num_nodes):
                    if i != j:
                        attention_sum += last_layer_attention[i, j].item()
                node_attention[i] = attention_sum
            
            # Normalize attention
            node_attention = node_attention / (node_attention.max() + 1e-8)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Graph with attention-colored nodes
            if node_positions is not None:
                pos = {i: (node_positions[i, 0].item(), node_positions[i, 1].item()) 
                      for i in range(num_nodes)}
            else:
                pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes with attention-based coloring
            node_colors = [node_attention[i].item() for i in range(num_nodes)]
            nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         cmap=plt.cm.Reds, node_size=300, ax=ax1)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax1)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, ax=ax1)
            
            ax1.set_title('Graph with Node Attention')
            ax1.axis('off')
            
            # Attention heatmap
            attention_matrix = last_layer_attention.cpu().numpy()
            im = ax2.imshow(attention_matrix, cmap='Blues', aspect='auto')
            ax2.set_xlabel('Target Nodes')
            ax2.set_ylabel('Source Nodes')
            ax2.set_title('Node Attention Matrix')
            plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            
            # Get top attended nodes
            top_node_indices = torch.argsort(node_attention, descending=True)[:5]
            top_nodes = [(int(idx.item()), node_attention[idx].item()) for idx in top_node_indices]
            
            return {
                'graph': G,
                'node_attention': node_attention.cpu().numpy(),
                'attention_matrix': attention_matrix,
                'top_nodes': top_nodes,
                'fig': fig,
                'statistics': {
                    'max_attention': float(torch.max(node_attention)),
                    'mean_attention': float(torch.mean(node_attention)),
                    'num_nodes': num_nodes
                }
            }
            
        except Exception as e:
            logger.error(f"Error visualizing graph attention: {str(e)}")
            return {}
    
    def visualize_cross_modal_attention(self, attention_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Visualize cross-modal attention between different modalities
        
        Args:
            attention_weights: Dictionary containing cross-modal attention weights
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            visualizations = {}
            
            # Image-Text attention
            if 'image_text_attention' in attention_weights:
                img_txt_attn = attention_weights['image_text_attention']
                if len(img_txt_attn.shape) == 4:  # [batch, heads, seq, seq]
                    attn = torch.mean(img_txt_attn[0], dim=0).cpu().numpy()
                else:  # [heads, seq, seq]
                    attn = torch.mean(img_txt_attn, dim=0).cpu().numpy()
                
                im1 = axes[0].imshow(attn, cmap='Reds', aspect='auto')
                axes[0].set_title('Image-Text Attention')
                axes[0].set_xlabel('Text Tokens')
                axes[0].set_ylabel('Image Patches')
                plt.colorbar(im1, ax=axes[0])
                
                visualizations['image_text'] = attn
            
            # Text-Graph attention
            if 'text_graph_attention' in attention_weights:
                txt_graph_attn = attention_weights['text_graph_attention']
                if len(txt_graph_attn.shape) == 4:
                    attn = torch.mean(txt_graph_attn[0], dim=0).cpu().numpy()
                else:
                    attn = torch.mean(txt_graph_attn, dim=0).cpu().numpy()
                
                im2 = axes[1].imshow(attn, cmap='Blues', aspect='auto')
                axes[1].set_title('Text-Graph Attention')
                axes[1].set_xlabel('Graph Nodes')
                axes[1].set_ylabel('Text Tokens')
                plt.colorbar(im2, ax=axes[1])
                
                visualizations['text_graph'] = attn
            
            # Graph-Image attention
            if 'graph_image_attention' in attention_weights:
                graph_img_attn = attention_weights['graph_image_attention']
                if len(graph_img_attn.shape) == 4:
                    attn = torch.mean(graph_img_attn[0], dim=0).cpu().numpy()
                else:
                    attn = torch.mean(graph_img_attn, dim=0).cpu().numpy()
                
                im3 = axes[2].imshow(attn, cmap='Greens', aspect='auto')
                axes[2].set_title('Graph-Image Attention')
                axes[2].set_xlabel('Image Patches')
                axes[2].set_ylabel('Graph Nodes')
                plt.colorbar(im3, ax=axes[2])
                
                visualizations['graph_image'] = attn
            
            # Overall attention summary
            if visualizations:
                # Calculate average attention across modalities
                all_attentions = list(visualizations.values())
                avg_attention = np.mean(all_attentions, axis=0)
                
                im4 = axes[3].imshow(avg_attention, cmap='viridis', aspect='auto')
                axes[3].set_title('Average Cross-Modal Attention')
                axes[3].set_xlabel('Target Modality')
                axes[3].set_ylabel('Source Modality')
                plt.colorbar(im4, ax=axes[3])
                
                visualizations['average'] = avg_attention
            
            plt.tight_layout()
            
            return {
                'visualizations': visualizations,
                'fig': fig,
                'attention_weights': attention_weights
            }
            
        except Exception as e:
            logger.error(f"Error visualizing cross-modal attention: {str(e)}")
            return {}
    
    def _resize_attention_map(self, attention_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize attention map to target size"""
        try:
            from scipy.ndimage import zoom
            
            current_size = attention_map.shape
            zoom_factors = (target_size[0] / current_size[0], target_size[1] / current_size[1])
            
            resized = zoom(attention_map, zoom_factors, order=1)
            return resized
            
        except ImportError:
            # Fallback to simple resizing
            import cv2
            return cv2.resize(attention_map, target_size)
    
    def _get_top_attention_patches(self, attention_map: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top attention patches"""
        try:
            patches = []
            h, w = attention_map.shape
            
            # Flatten and get top indices
            flat_attention = attention_map.flatten()
            top_indices = np.argsort(flat_attention)[-top_k:][::-1]
            
            for idx in top_indices:
                row = idx // w
                col = idx % w
                attention_score = flat_attention[idx]
                
                patches.append({
                    'row': row,
                    'col': col,
                    'attention_score': attention_score,
                    'bbox': (col, row, 1, 1)  # Single patch
                })
            
            return patches
            
        except Exception as e:
            logger.error(f"Error getting top attention patches: {str(e)}")
            return []
    
    def _highlight_patches(self, ax, patches: List[Dict[str, Any]], patches_per_dim: int):
        """Highlight top attention patches on the plot"""
        try:
            for patch in patches:
                row, col = patch['row'], patch['col']
                rect = patches.Rectangle((col, row), 1, 1, linewidth=2, 
                                       edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
                
        except Exception as e:
            logger.error(f"Error highlighting patches: {str(e)}")
    
    def save_visualization(self, visualization: Dict[str, Any], save_path: str):
        """Save attention visualization"""
        try:
            if 'fig' in visualization:
                visualization['fig'].savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
                plt.close(visualization['fig'])
            
            # Save metadata
            metadata = {
                'statistics': visualization.get('statistics', {}),
                'top_patches': visualization.get('top_patches', []),
                'top_tokens': visualization.get('top_tokens', []),
                'top_nodes': visualization.get('top_nodes', [])
            }
            
            with open(f"{save_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Visualization saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
