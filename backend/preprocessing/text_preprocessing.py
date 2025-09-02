"""
Text preprocessing module for natural language captions and queries
"""

import re
import string
import nltk
import spacy
from typing import List, Dict, Any, Optional
import logging
from transformers import DistilBertTokenizer
import torch

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing for natural language processing"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize stopwords
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Floor plan specific vocabulary
        self.floorplan_vocab = {
            'room_types': ['bedroom', 'bathroom', 'kitchen', 'living_room', 'dining_room', 
                          'closet', 'hallway', 'office', 'study', 'garage', 'basement'],
            'spatial_relations': ['next_to', 'adjacent', 'connected', 'near', 'far_from', 
                                 'above', 'below', 'left_of', 'right_of', 'inside', 'outside'],
            'architectural_elements': ['door', 'window', 'wall', 'corridor', 'entrance', 
                                     'exit', 'staircase', 'elevator', 'balcony', 'patio'],
            'measurements': ['square_feet', 'sqft', 'meters', 'feet', 'inches', 'large', 
                           'small', 'big', 'tiny', 'huge', 'compact']
        }
    
    def preprocess_text(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Preprocess text for model input
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with preprocessed text data
        """
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Tokenize
            tokens = self._tokenize(cleaned_text)
            
            # Extract features
            features = self._extract_text_features(cleaned_text)
            
            # Create model input
            model_input = self._create_model_input(cleaned_text, max_length)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'features': features,
                'model_input': model_input,
                'length': len(tokens)
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,!?;:-]', '', text)
            
            # Remove multiple punctuation
            text = re.sub(r'[.,!?;:]{2,}', '.', text)
            
            # Strip whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            # Use NLTK for basic tokenization
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords and punctuation
            tokens = [token for token in tokens 
                     if token not in self.stopwords and 
                     token not in string.punctuation]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            return text.split()
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text"""
        try:
            features = {}
            
            # Basic statistics
            features['word_count'] = len(text.split())
            features['char_count'] = len(text)
            features['sentence_count'] = len(nltk.sent_tokenize(text))
            
            # Floor plan specific features
            features['contains_room_types'] = self._contains_room_types(text)
            features['contains_spatial_relations'] = self._contains_spatial_relations(text)
            features['contains_architectural_elements'] = self._contains_architectural_elements(text)
            features['contains_measurements'] = self._contains_measurements(text)
            
            # Named entity recognition (if spaCy is available)
            if self.nlp:
                doc = self.nlp(text)
                features['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
                features['pos_tags'] = [token.pos_ for token in doc]
            else:
                features['entities'] = []
                features['pos_tags'] = []
            
            # Sentiment analysis (simple rule-based)
            features['sentiment'] = self._analyze_sentiment(text)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}")
            return {}
    
    def _contains_room_types(self, text: str) -> List[str]:
        """Check if text contains room type mentions"""
        found_types = []
        for room_type in self.floorplan_vocab['room_types']:
            if room_type in text:
                found_types.append(room_type)
        return found_types
    
    def _contains_spatial_relations(self, text: str) -> List[str]:
        """Check if text contains spatial relation mentions"""
        found_relations = []
        for relation in self.floorplan_vocab['spatial_relations']:
            if relation in text:
                found_relations.append(relation)
        return found_relations
    
    def _contains_architectural_elements(self, text: str) -> List[str]:
        """Check if text contains architectural element mentions"""
        found_elements = []
        for element in self.floorplan_vocab['architectural_elements']:
            if element in text:
                found_elements.append(element)
        return found_elements
    
    def _contains_measurements(self, text: str) -> List[str]:
        """Check if text contains measurement mentions"""
        found_measurements = []
        for measurement in self.floorplan_vocab['measurements']:
            if measurement in text:
                found_measurements.append(measurement)
        return found_measurements
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'nice', 'beautiful', 'spacious', 'large', 'comfortable']
        negative_words = ['bad', 'small', 'cramped', 'dark', 'narrow', 'uncomfortable']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _create_model_input(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        """Create model input tensors"""
        try:
            # Tokenize with DistilBERT tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            }
            
        except Exception as e:
            logger.error(f"Error creating model input: {str(e)}")
            raise
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess user query for VQA"""
        try:
            # Preprocess as regular text
            result = self.preprocess_text(query, max_length=128)
            
            # Add query-specific features
            result['query_type'] = self._classify_query_type(query)
            result['expected_answer_type'] = self._get_expected_answer_type(query)
            
            return result
            
        except Exception as e:
            logger.error(f"Error preprocessing query: {str(e)}")
            raise
    
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
    
    def _get_expected_answer_type(self, query: str) -> str:
        """Get expected answer type for the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how many', 'count', 'number']):
            return 'number'
        elif any(word in query_lower for word in ['is', 'are', 'does', 'do']):
            return 'yes_no'
        elif any(word in query_lower for word in ['what color', 'color']):
            return 'color'
        elif any(word in query_lower for word in ['where', 'location', 'position']):
            return 'location'
        else:
            return 'text'
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text"""
        try:
            tokens = self._tokenize(text)
            
            # Count word frequencies
            word_freq = {}
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Return top k words
            return [word for word, freq in sorted_words[:top_k]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
