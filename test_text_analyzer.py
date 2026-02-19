"""
Test Text Emotion Analyzer
Uses all 28 emotions from GoEmotions dataset
"""

import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
interpreter = tf.lite.Interpreter(model_path='emotion_text_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Load emotions
with open('go_emotions_dataset.csv', 'r') as f:
    emotions = json.load(f)

def analyze_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post').astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    emotion_idx = np.argmax(output)
    return emotions[emotion_idx], float(output[emotion_idx])

# Test
print("Text Emotion Analyzer - Type 'quit' to exit\n")

while True:
    text = input("Enter text: ")
    if text.lower() == 'quit':
        break
    
    emotion, confidence = analyze_text(text)
    print(f"Emotion: {emotion} (Confidence: {confidence*100:.1f}%)\n")
