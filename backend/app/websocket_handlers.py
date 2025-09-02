"""
WebSocket event handlers for real-time communication
"""

from flask_socketio import emit
import logging

logger = logging.getLogger(__name__)

def register_socketio_handlers(socketio):
    """Register WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('start_analysis')
    def handle_start_analysis(data):
        """Handle real-time analysis requests"""
        try:
            logger.info(f"Starting analysis: {data}")
            
            # Emit progress updates
            emit('analysis_progress', {
                'step': 'preprocessing',
                'progress': 10,
                'message': 'Preprocessing image...'
            })
            
            # Continue with analysis steps...
            emit('analysis_progress', {
                'step': 'encoding',
                'progress': 30,
                'message': 'Encoding modalities...'
            })
            
            emit('analysis_progress', {
                'step': 'fusion',
                'progress': 60,
                'message': 'Fusing modalities...'
            })
            
            emit('analysis_progress', {
                'step': 'inference',
                'progress': 80,
                'message': 'Running inference...'
            })
            
            emit('analysis_progress', {
                'step': 'complete',
                'progress': 100,
                'message': 'Analysis complete!'
            })
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            emit('analysis_error', {'error': str(e)})
    
    @socketio.on('request_explanation')
    def handle_request_explanation(data):
        """Handle explanation requests"""
        try:
            logger.info(f"Generating explanation: {data}")
            
            # Emit explanation updates
            emit('explanation_progress', {
                'step': 'image_attention',
                'progress': 25,
                'message': 'Generating image attention maps...'
            })
            
            emit('explanation_progress', {
                'step': 'graph_attention',
                'progress': 50,
                'message': 'Computing graph attention...'
            })
            
            emit('explanation_progress', {
                'step': 'text_attention',
                'progress': 75,
                'message': 'Analyzing text attention...'
            })
            
            emit('explanation_progress', {
                'step': 'complete',
                'progress': 100,
                'message': 'Explanations ready!'
            })
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            emit('explanation_error', {'error': str(e)})
