# Setup Guide

This guide will help you set up the Multi-Modal Floor Plan Understanding system on your local machine.

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **CUDA**: 11.8 or higher (optional, for GPU acceleration)
- **RAM**: 8GB or more recommended
- **Disk Space**: 10GB or more for models and dependencies

### Software Dependencies

- Git
- pip (Python package manager)
- npm (Node.js package manager)

## Quick Start

### 1. Clone the Repository

```bash
git clone [repository-url]
cd Floore
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start the backend server
python app/main.py
```

The backend server will start on `http://localhost:5000`.

### 3. Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will start on `http://localhost:3000`.

## Detailed Setup

### Backend Configuration

#### Environment Variables

Create a `.env` file in the backend directory:

```env
# Flask Configuration
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Model Configuration
DEVICE=cuda  # or cpu
MODEL_CACHE_DIR=./models/cache

# Dataset Configuration
DATASET_PATH=./data
PROCESSED_DATA_PATH=./data/processed

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

#### Model Downloads

The system will automatically download pre-trained models on first run:

- **Vision Transformer**: ViT-Base-Patch16-224
- **DistilBERT**: distilbert-base-uncased
- **spaCy**: en_core_web_sm

### Frontend Configuration

#### Environment Variables

Create a `.env` file in the frontend directory:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000

# Development
REACT_APP_DEBUG=true
```

## Usage

### 1. Access the Application

Open your web browser and navigate to `http://localhost:3000`.

### 2. Upload a Floor Plan

1. Click on "Start Analysis" or navigate to the Analysis page
2. Upload a floor plan image (PNG, JPG, JPEG, GIF, BMP, TIFF)
3. Choose the analysis type:
   - **Full Analysis**: Complete multi-modal analysis
   - **VQA**: Visual Question Answering
   - **Retrieval**: Similarity search
   - **Validity**: Constraint validation

### 3. View Results

- **Results Tab**: View analysis results and confidence scores
- **Visualization Tab**: Interactive visualizations of attention maps and graph structures
- **Explainability Tab**: Detailed explanations of AI decision-making

## Troubleshooting

### Common Issues

#### Backend Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**CUDA Issues**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA is not available, set device to CPU
export DEVICE=cpu
```

**Model Download Issues**
```bash
# Clear model cache and retry
rm -rf models/cache/
python app/main.py
```

#### Frontend Issues

**Dependencies Issues**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Connection Issues**
- Ensure backend server is running on port 5000
- Check firewall settings
- Verify WebSocket connection in browser developer tools

### Performance Optimization

#### GPU Acceleration

To enable GPU acceleration:

1. Install CUDA toolkit
2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Set environment variable:
```bash
export DEVICE=cuda
```

#### Memory Optimization

For systems with limited RAM:

1. Reduce batch size in model configurations
2. Use CPU instead of GPU
3. Process smaller images
4. Enable model quantization

## Development

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Quality

```bash
# Backend linting
cd backend
black .
flake8 .

# Frontend linting
cd frontend
npm run lint
```

### Building for Production

```bash
# Frontend build
cd frontend
npm run build

# Backend deployment
cd backend
gunicorn app.main:app
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the documentation in the `/docs` directory
3. Check GitHub issues for known problems
4. Create a new issue with detailed error information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
