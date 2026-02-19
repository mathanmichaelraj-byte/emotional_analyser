"""
Text Emotion Model Training
Uses GoEmotions dataset emotions directly
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('go_emotions_dataset.csv')

# Get all emotion columns (excluding metadata)
emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Get primary emotion for each text
def get_primary_emotion(row):
    for i, col in enumerate(emotion_cols):
        if row[col] == 1:
            return i
    return 27  # neutral as default

df['label'] = df.apply(get_primary_emotion, axis=1)

print(f"Dataset: {len(df)} samples")
print(f"Emotions: {len(emotion_cols)}")
print(f"Distribution:\n{df['label'].value_counts()}")

# Tokenize
texts = df['text'].values
labels = df['label'].values

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

y_categorical = tf.keras.utils.to_categorical(labels, len(emotion_cols))

# Build model (simpler architecture for TFLite compatibility)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 64, input_length=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(emotion_cols), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train
X_train, X_val, y_train, y_val = train_test_split(padded, y_categorical, test_size=0.2, random_state=42, stratify=labels)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
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

# Save tokenizer and emotion list
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

with open('emotions.json', 'w') as f:
    json.dump(emotion_cols, f)

print(f"Model saved: emotion_text_model.tflite ({len(tflite_model)/1024:.2f} KB)")
print(f"Emotions: {emotion_cols}")
