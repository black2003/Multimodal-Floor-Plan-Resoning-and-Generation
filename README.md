# Multi-Modal Floor Plan Understanding and Reasoning

A comprehensive AI research and application project for multi-modal floor-plan understanding and reasoning, featuring deep learning modules, preprocessing, fusion, explainability, and a complete frontend interface.

## Architecture Overview

This project implements an end-to-end pipeline for floor plan analysis with the following components:

- **Preprocessing**: Image normalization, classical segmentation, text tokenization
- **Graph Builder**: Room nodes with geometry, adjacency edges with door/hall connectivity
- **Encoders**: ViT (image), DistilBERT (text), GNN (graph)
- **Cross-Modal Fusion**: Transformer-based fusion of multi-modal embeddings
- **Multi-Task Heads**: VQA, Retrieval, Validity/Repair
- **Explainability**: Grad-CAM, attention visualization, constraint tracing
- **Frontend**: React-based interactive interface

##  Project Structure

```
├── backend/                 # Flask/FastAPI backend
│   ├── app/                # Main application
│   ├── models/             # Deep learning models
│   ├── preprocessing/      # Data preprocessing modules
│   ├── explainability/     # Explainability modules
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Node dependencies
├── data/                  # Dataset and processed data
└── docs/                  # Documentation
```

##  Quick Start

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python app/main.py
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

##  Features

- **Multi-Modal Analysis**: Process images, text, and graph representations
- **Interactive Visualization**: Real-time attention maps and graph visualization
- **Explainable AI**: Comprehensive interpretability across all modalities
- **Question Answering**: Natural language queries about floor plans
- **Similarity Retrieval**: Find similar floor plans using contrastive learning
- **Validity Checking**: Neuro-symbolic validation with repair suggestions

##  Dataset

Uses the pseudo-floor-plan-12k dataset containing floor plan images and natural language captions.

##  Tech Stack

- **Backend**: Flask/FastAPI, PyTorch, Transformers
- **Frontend**: React, TailwindCSS, shadcn/ui
- **ML Models**: ViT, DistilBERT, Graph Attention Networks
- **Visualization**: D3.js, WebGL, Canvas API

##  License

MIT License - see LICENSE file for details.
