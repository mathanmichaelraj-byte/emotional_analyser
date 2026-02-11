"""
Text-based Emotion Classifier
Analyzes user text input for emotional content
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('go_emotions_dataset.csv')

# Map to 6 emotions
emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                'relief', 'remorse', 'sadness', 'surprise', 'neutral']

def map_to_6_emotions(row):
    if row['neutral'] == 1: return 4
    if any(row[e] == 1 for e in ['relief', 'approval', 'gratitude', 'caring', 'optimism']): return 0  # calm
    if any(row[e] == 1 for e in ['nervousness', 'confusion', 'curiosity', 'surprise']): return 1  # restless
    if any(row[e] == 1 for e in ['fear', 'annoyance', 'disappointment']): return 2  # stressed
    if any(row[e] == 1 for e in ['sadness', 'grief', 'embarrassment']): return 3  # low_energy
    if any(row[e] == 1 for e in ['anger', 'disgust', 'remorse']): return 5  # distressed
    return 4

df['emotion_label'] = df.apply(map_to_6_emotions, axis=1)

# Prepare text data
texts = df['text'].values
labels = df['emotion_label'].values

# Tokenize
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

y_categorical = tf.keras.utils.to_categorical(labels, 6)

# Build model (deeper architecture for better accuracy)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=50),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
X_train, X_val, y_train, y_val = train_test_split(padded, y_categorical, test_size=0.2, random_state=42)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Evaluate
val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
print(f"\nValidation accuracy: {val_acc*100:.2f}%")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('emotion_text_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save tokenizer
import json
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer_json, f)

print(f"\nText model saved: emotion_text_model.tflite ({len(tflite_model)/1024:.2f} KB)")
