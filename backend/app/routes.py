"""
API routes for the Multi-Modal Floor Plan Understanding application
"""

from flask import Blueprint, request, jsonify, current_app
import logging
import base64
import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Multi-Modal Floor Plan Understanding API is running'
    })

@api_bp.route('/upload', methods=['POST'])
def upload_floorplan():
    """Upload and process floor plan image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image = Image.open(file.stream)
        
        # Get pipeline from app context
        pipeline = current_app.pipeline
        
        # Process the image
        result = pipeline.process_image(image)
        
        return jsonify({
            'success': True,
            'result': result,
            'message': 'Floor plan processed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analyze', methods=['POST'])
def analyze_floorplan():
    """Analyze floor plan with text query"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        image_data = data.get('image')
        query = data.get('query', '')
        analysis_type = data.get('type', 'full')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Get pipeline from app context
        pipeline = current_app.pipeline
        
        # Process the analysis
        result = pipeline.process_floorplan(
            image=image,
            text_query=query,
            analysis_type=analysis_type
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error analyzing floor plan: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/vqa', methods=['POST'])
def visual_question_answering():
    """Visual Question Answering endpoint"""
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        question = data.get('question', '')
        
        if not image_data or not question:
            return jsonify({'error': 'Image and question are required'}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get pipeline and process VQA
        pipeline = current_app.pipeline
        result = pipeline.answer_question(image, question)
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'explanation': result.get('explanation', '')
        })
        
    except Exception as e:
        logger.error(f"Error in VQA: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/retrieve', methods=['POST'])
def retrieve_similar():
    """Retrieve similar floor plans"""
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        top_k = data.get('top_k', 5)
        
        if not image_data:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get pipeline and retrieve similar plans
        pipeline = current_app.pipeline
        result = pipeline.retrieve_similar(image, top_k=top_k)
        
        return jsonify({
            'success': True,
            'similar_plans': result['similar_plans'],
            'similarities': result['similarities']
        })
        
    except Exception as e:
        logger.error(f"Error retrieving similar plans: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/validate', methods=['POST'])
def validate_floorplan():
    """Validate floor plan structure and suggest repairs"""
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get pipeline and validate
        pipeline = current_app.pipeline
        result = pipeline.validate_floorplan(image)
        
        return jsonify({
            'success': True,
            'validity_score': result['validity_score'],
            'issues': result['issues'],
            'suggestions': result['suggestions']
        })
        
    except Exception as e:
        logger.error(f"Error validating floor plan: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/explain', methods=['POST'])
def explain_analysis():
    """Get explainability information for analysis"""
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        analysis_id = data.get('analysis_id')
        modality = data.get('modality', 'all')  # 'image', 'text', 'graph', 'all'
        
        if not image_data:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get pipeline and generate explanations
        pipeline = current_app.pipeline
        result = pipeline.explain_analysis(image, analysis_id, modality)
        
        return jsonify({
            'success': True,
            'explanations': result
        })
        
    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        return jsonify({'error': str(e)}), 500
