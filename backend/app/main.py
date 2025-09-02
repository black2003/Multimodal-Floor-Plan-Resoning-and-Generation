"""
Main Flask application for Multi-Modal Floor Plan Understanding and Reasoning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import logging
from datetime import datetime

# Import our modules
from .routes import api_bp
from .websocket_handlers import socketio_handlers
from ..models.multimodal_pipeline import MultiModalPipeline
from ..utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialize the multi-modal pipeline
    pipeline = MultiModalPipeline()
    
    # Store pipeline in app context
    app.pipeline = pipeline
    
    # WebSocket event handlers
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to server', 'timestamp': datetime.now().isoformat()})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('analyze_floorplan')
    def handle_analyze_floorplan(data):
        """Handle floor plan analysis requests via WebSocket"""
        try:
            logger.info(f"Received analysis request: {data}")
            
            # Emit progress updates
            emit('analysis_progress', {'step': 'preprocessing', 'progress': 10})
            
            # Process the request
            result = pipeline.process_floorplan(
                image_data=data.get('image'),
                text_query=data.get('query', ''),
                analysis_type=data.get('type', 'full')
            )
            
            emit('analysis_progress', {'step': 'complete', 'progress': 100})
            emit('analysis_result', result)
            
        except Exception as e:
            logger.error(f"Error processing floor plan: {str(e)}")
            emit('analysis_error', {'error': str(e)})
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    
    # Run the application
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        allow_unsafe_werkzeug=True
    )
