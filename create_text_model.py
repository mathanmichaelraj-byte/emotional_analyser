"""
Text Emotion Classifier - GoEmotions Dataset
Classifies text into 6 primary emotion categories
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('go_emotions_dataset.csv')

# Use top 6 emotions from GoEmotions
emotion_cols = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise']

# Create single label (pick dominant emotion)
def get_primary_emotion(row):
    for i, col in enumerate(emotion_cols):
        if row[col] == 1:
            return i
    return 0  # default to neutral

df['emotion_label'] = df.apply(get_primary_emotion, axis=1)

# Filter to only include samples with our 6 emotions
df = df[df['emotion_label'].isin([0, 1, 2, 3, 4, 5])]

print(f"Dataset size: {len(df)}")
print(f"Emotion distribution:\n{df['emotion_label'].value_counts()}")

# Prepare text data
texts = df['text'].values
labels = df['emotion_label'].values

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

y_categorical = tf.keras.utils.to_categorical(labels, 6)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=50),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
X_train, X_val, y_train, y_val = train_test_split(
    padded, y_categorical, test_size=0.2, random_state=42, stratify=labels
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    ],
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
print(f"Emotions: {emotion_cols}")
