# System Architecture

This document describes the architecture of the Multi-Modal Floor Plan Understanding system.

## Overview

The system implements an end-to-end pipeline for multi-modal floor plan analysis, combining computer vision, natural language processing, graph neural networks, and explainable AI techniques.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Models        │
│   (React)       │◄──►│   (Flask)       │◄──►│   (PyTorch)     │
│                 │    │                 │    │                 │
│ • Upload UI     │    │ • API Routes    │    │ • ViT Encoder   │
│ • Visualization │    │ • WebSockets    │    │ • DistilBERT    │
│ • Results       │    │ • Pipeline      │    │ • GNN Encoder   │
│ • Explainability│    │ • Explainability│    │ • Fusion        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Backend Architecture

### Core Components

#### 1. Preprocessing Module
- **Image Preprocessing**: Normalization, resizing, enhancement
- **Classical Segmentation**: Walls, rooms, doors detection using OpenCV
- **Text Preprocessing**: Tokenization, feature extraction using NLTK/spaCy
- **Graph Builder**: Convert segmentation to graph representation

#### 2. Encoder Module
- **Vision Transformer (ViT)**: Image feature extraction
- **DistilBERT**: Text feature extraction
- **Graph Neural Network**: Graph feature extraction using GAT/GraphSAGE

#### 3. Fusion Module
- **Cross-Modal Fusion Transformer**: Integrate multi-modal embeddings
- **Attention Mechanisms**: Multi-head attention across modalities
- **Feature Alignment**: Align features from different modalities

#### 4. Task Heads
- **VQA Head**: Visual Question Answering with confidence estimation
- **Retrieval Head**: Contrastive learning for similarity search
- **Validity Head**: Neuro-symbolic validation with constraint checking

#### 5. Explainability Module
- **Grad-CAM**: Visual attention explanation
- **Attention Visualization**: Multi-modal attention maps
- **Constraint Tracer**: Rule-based validation explanations

### Data Flow

```
Raw Image + Text Query
        ↓
   Preprocessing
        ↓
┌───────┼───────┐
│       │       │
ViT   DistilBERT  GNN
│       │       │
└───────┼───────┘
        ↓
   Fusion Transformer
        ↓
┌───────┼───────┐
│       │       │
VQA   Retrieval  Validity
│       │       │
└───────┼───────┘
        ↓
   Explainability
        ↓
   Final Results
```

## Frontend Architecture

### Component Structure

```
App
├── Layout
│   ├── Navigation
│   ├── ThemeToggle
│   └── ConnectionStatus
├── Pages
│   ├── HomePage
│   ├── AnalysisPage
│   ├── ExplainabilityPage
│   └── DocumentationPage
├── Components
│   ├── ImageUploader
│   ├── QueryInput
│   ├── ResultsDisplay
│   └── VisualizationPanel
└── Contexts
    ├── ThemeContext
    ├── SocketContext
    └── AnalysisContext
```

### State Management

- **React Context**: Global state management
- **Local State**: Component-specific state
- **WebSocket**: Real-time communication
- **Local Storage**: User preferences and history

## Model Architecture

### Vision Transformer (ViT)

```python
class ImageEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224"):
        self.vit = timm.create_model(model_name, pretrained=True)
        self.projection = nn.Linear(feature_dim, output_dim)
    
    def forward(self, images):
        features = self.vit(images)
        return self.projection(features)
```

### DistilBERT Text Encoder

```python
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return self.projection(outputs.last_hidden_state[:, 0])
```

### Graph Neural Network

```python
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.gat_layers = nn.ModuleList([
            GATConv(input_dim, hidden_dim, heads=8)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, graph_data):
        x = graph_data.x
        for layer in self.gat_layers:
            x = layer(x, graph_data.edge_index)
        return global_mean_pool(x, graph_data.batch)
```

### Cross-Modal Fusion

```python
class CrossModalFusionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, image_features, text_features, graph_features):
        for layer in self.fusion_layers:
            image_features, text_features, graph_features = layer(
                image_features, text_features, graph_features
            )
        return torch.cat([image_features, text_features, graph_features], dim=-1)
```

## API Architecture

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload floor plan image |
| POST | `/api/analyze` | Analyze floor plan with query |
| POST | `/api/vqa` | Visual Question Answering |
| POST | `/api/retrieve` | Retrieve similar plans |
| POST | `/api/validate` | Validate floor plan |
| POST | `/api/explain` | Get explanations |
| GET | `/api/health` | Health check |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `analyze_floorplan` | Client → Server | Start analysis |
| `analysis_progress` | Server → Client | Progress updates |
| `analysis_result` | Server → Client | Final results |
| `analysis_error` | Server → Client | Error messages |
| `request_explanation` | Client → Server | Request explanations |

## Data Models

### Analysis Result

```typescript
interface AnalysisResult {
  id: string;
  timestamp: string;
  image: string;
  query?: string;
  vqa?: {
    answer: string;
    confidence: number;
    top_answers: Array<{text: string; probability: number}>;
  };
  retrieval?: {
    similar_plans: Array<{id: number; similarity: number; description: string}>;
    similarities: number[];
  };
  validity?: {
    validity_score: number;
    issues: any;
    suggestions: any;
    constraint_scores: Record<string, number>;
  };
  metadata: {
    image_shape: [number, number];
    analysis_type: string;
    segmentation_info: {
      num_rooms: number;
      num_doors: number;
      num_walls: number;
    };
  };
}
```

### Graph Data

```python
@dataclass
class GraphData:
    x: torch.Tensor  # Node features
    edge_index: torch.Tensor  # Edge indices
    edge_attr: torch.Tensor  # Edge features
    node_types: List[str]  # Node type labels
    edge_types: List[str]  # Edge type labels
    num_nodes: int
    num_edges: int
```

## Performance Considerations

### Optimization Strategies

1. **Model Optimization**
   - Model quantization for inference
   - Batch processing for multiple requests
   - Caching of preprocessed features

2. **Memory Management**
   - Gradient checkpointing
   - Dynamic batching
   - Model sharding

3. **Scalability**
   - Horizontal scaling with load balancers
   - Database optimization
   - CDN for static assets

### Monitoring

- **Metrics**: Response time, throughput, error rates
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Automated health monitoring
- **Alerting**: Real-time alerting for critical issues

## Security Considerations

### Data Protection

- **Input Validation**: Sanitize all user inputs
- **File Upload Security**: Validate file types and sizes
- **Rate Limiting**: Prevent abuse and DoS attacks
- **CORS Configuration**: Proper cross-origin resource sharing

### Model Security

- **Model Integrity**: Verify model checksums
- **Inference Security**: Protect against adversarial attacks
- **Data Privacy**: No persistent storage of user data
- **Access Control**: Authentication and authorization

## Deployment Architecture

### Development Environment

```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │
│   (localhost:3000)│◄──►│   (localhost:5000)│
│   React Dev     │    │   Flask Dev     │
└─────────────────┘    └─────────────────┘
```

### Production Environment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Servers   │    │   GPU Servers   │
│   (Nginx)       │◄──►│   (Gunicorn)    │◄──►│   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CDN           │    │   Database      │    │   Model Cache   │
│   (Static Files)│    │   (PostgreSQL)  │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Future Enhancements

### Planned Features

1. **Advanced Models**
   - Larger transformer models
   - Specialized architectural models
   - Multi-scale feature extraction

2. **Enhanced Explainability**
   - Counterfactual explanations
   - Causal reasoning
   - Interactive explanation editing

3. **Performance Improvements**
   - Model compression
   - Faster inference
   - Real-time processing

4. **Extended Capabilities**
   - 3D floor plan support
   - Multi-language support
   - Advanced constraint checking

### Research Directions

1. **Novel Architectures**
   - Cross-modal attention mechanisms
   - Hierarchical graph representations
   - Multi-task learning frameworks

2. **Explainability Research**
   - Interpretable AI methods
   - Human-AI collaboration
   - Trust and transparency

3. **Domain Applications**
   - Architectural design assistance
   - Building code compliance
   - Accessibility analysis
