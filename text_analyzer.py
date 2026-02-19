"""
Text Emotion Analyzer
Analyzes user text input for emotions
"""

import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextEmotionAnalyzer:
    def __init__(self, model_path='emotion_text_model.tflite', tokenizer_path='tokenizer.json'):
        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load tokenizer
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tokenizer_from_json(tokenizer_json)
        
        self.emotions = ['Neutral', 'Joy', 'Sadness', 'Anger', 'Fear', 'Surprise']
    
    def analyze(self, text):
        """
        Analyze text for emotional content
        text: string input from user
        """
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
        padded = padded.astype(np.float32)
        
        # Predict
        self.interpreter.set_tensor(self.input_details[0]['index'], padded)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        emotion_idx = np.argmax(output)
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': float(output[emotion_idx]),
            'probabilities': {self.emotions[i]: float(output[i]) for i in range(6)}
        }

if __name__ == '__main__':
    analyzer = TextEmotionAnalyzer()
    
    # Example texts
    texts = [
        "I'm feeling great today!",
        "This is so frustrating and annoying",
        "I'm really worried about tomorrow"
    ]
    
    for text in texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
