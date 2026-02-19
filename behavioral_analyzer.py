"""
Behavioral Emotion Analyzer
Analyzes user behavioral patterns
"""

import numpy as np
import tensorflow as tf

class BehavioralEmotionAnalyzer:
    def __init__(self, model_path='emotion_model.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.emotions = ['Calm', 'Restless', 'Stressed', 'Low Energy', 'Neutral', 'Distressed']
    
    def analyze(self, features):
        """
        Analyze behavioral patterns
        features: list of 20 behavioral metrics
        """
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        emotion_idx = np.argmax(output)
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': float(output[emotion_idx]),
            'probabilities': {self.emotions[i]: float(output[i]) for i in range(6)}
        }

if __name__ == '__main__':
    analyzer = BehavioralEmotionAnalyzer()
    
    # Example: Calm pattern
    features = [5, 200, 3, 0.1, 0.4, 0.3, 0.3, 0.7, 0.2, 0.3, 100, 0.1, 0.6, 2, 10, 3, 6, 6, 0.1, 80]
    result = analyzer.analyze(features)
    print(f"Result: {result}")
