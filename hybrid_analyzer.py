"""
Hybrid Emotion Analysis System
Combines behavioral patterns + text analysis
"""

import numpy as np
import tensorflow as tf

class HybridEmotionAnalyzer:
    def __init__(self):
        # Load behavioral model
        self.behavior_interpreter = tf.lite.Interpreter(model_path='emotion_model.tflite')
        self.behavior_interpreter.allocate_tensors()
        self.behavior_input = self.behavior_interpreter.get_input_details()
        self.behavior_output = self.behavior_interpreter.get_output_details()
        
        # Load text model (if exists)
        try:
            self.text_interpreter = tf.lite.Interpreter(model_path='emotion_text_model.tflite')
            self.text_interpreter.allocate_tensors()
            self.text_input = self.text_interpreter.get_input_details()
            self.text_output = self.text_interpreter.get_output_details()
            self.has_text_model = True
        except:
            self.has_text_model = False
            print("Text model not found, using behavioral only")
        
        self.emotions = ['Calm', 'Restless', 'Stressed', 'Low Energy', 'Neutral', 'Distressed']
    
    def analyze_behavior(self, features):
        """Analyze behavioral patterns (20 features)"""
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        self.behavior_interpreter.set_tensor(self.behavior_input[0]['index'], features)
        self.behavior_interpreter.invoke()
        output = self.behavior_interpreter.get_tensor(self.behavior_output[0]['index'])
        return output[0]
    
    def analyze_text(self, text_sequence):
        """Analyze text input (preprocessed sequence)"""
        if not self.has_text_model:
            return None
        
        text_sequence = np.array(text_sequence, dtype=np.int32).reshape(1, -1)
        self.text_interpreter.set_tensor(self.text_input[0]['index'], text_sequence)
        self.text_interpreter.invoke()
        output = self.text_interpreter.get_tensor(self.text_output[0]['index'])
        return output[0]
    
    def hybrid_analysis(self, behavioral_features=None, text_sequence=None, behavior_weight=0.6):
        """
        Combine both analyses
        behavior_weight: 0.6 means 60% behavioral, 40% text
        """
        predictions = []
        
        if behavioral_features is not None:
            behavior_pred = self.analyze_behavior(behavioral_features)
            predictions.append((behavior_pred, behavior_weight))
        
        if text_sequence is not None and self.has_text_model:
            text_pred = self.analyze_text(text_sequence)
            predictions.append((text_pred, 1 - behavior_weight))
        
        if not predictions:
            return None
        
        # Weighted average
        final_pred = np.zeros(6)
        for pred, weight in predictions:
            final_pred += pred * weight
        
        emotion_idx = np.argmax(final_pred)
        confidence = final_pred[emotion_idx]
        
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': float(confidence),
            'probabilities': {self.emotions[i]: float(final_pred[i]) for i in range(6)}
        }

# Example usage
if __name__ == '__main__':
    analyzer = HybridEmotionAnalyzer()
    
    # Example: Behavioral features only
    behavior_features = [5, 200, 3, 0.1, 0.4, 0.3, 0.3, 0.7, 0.2, 0.3, 100, 0.1, 0.6, 2, 10, 3, 6, 6, 0.1, 80]
    result = analyzer.hybrid_analysis(behavioral_features=behavior_features)
    print(f"Behavioral only: {result}")
    
    # Example: Both (text sequence would come from tokenizer)
    # result = analyzer.hybrid_analysis(behavioral_features=behavior_features, text_sequence=text_seq)
