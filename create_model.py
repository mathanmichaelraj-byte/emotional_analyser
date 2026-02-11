"""
Emotion Classifier - TFLite Model Generator
Generates a neural network model for classifying 6 emotional states
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_emotional_data(n_samples=2000):
    """Generate synthetic training data for 6 emotional states"""
    X, y = [], []
    
    # States: 0=calm, 1=restless, 2=stressed, 3=lowEnergy, 4=neutral, 5=distressed
    for _ in range(n_samples):
        state = np.random.randint(0, 6)
        
        if state == 0:  # Calm
            features = [
                np.random.uniform(3, 6), np.random.uniform(120, 300), np.random.uniform(2, 4),
                np.random.uniform(0, 0.2), np.random.uniform(0.3, 0.5), np.random.uniform(0.2, 0.4),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.1, 0.3),
                np.random.uniform(0.2, 0.4), np.random.uniform(50, 150), np.random.uniform(0, 0.2),
                np.random.uniform(0.5, 0.8), np.random.uniform(1, 3), np.random.uniform(5, 15),
                np.random.uniform(2, 5), np.random.uniform(5, 7), np.random.uniform(5, 7),
                np.random.uniform(0, 0.2), np.random.uniform(40, 120)
            ]
        elif state == 1:  # Restless
            features = [
                np.random.uniform(10, 15), np.random.uniform(30, 120), np.random.uniform(6, 10),
                np.random.uniform(0.3, 0.5), np.random.uniform(0.1, 0.3), np.random.uniform(0.3, 0.5),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8),
                np.random.uniform(0.1, 0.2), np.random.uniform(200, 400), np.random.uniform(0.4, 0.7),
                np.random.uniform(0.1, 0.3), np.random.uniform(2, 5), np.random.uniform(20, 40),
                np.random.uniform(8, 15), np.random.uniform(5, 7), np.random.uniform(5, 7),
                np.random.uniform(0.5, 0.7), np.random.uniform(200, 350)
            ]
        elif state == 2:  # Stressed
            features = [
                np.random.uniform(8, 12), np.random.uniform(200, 400), np.random.uniform(4, 7),
                np.random.uniform(0.6, 0.8), np.random.uniform(0.1, 0.2), np.random.uniform(0.4, 0.6),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.3, 0.5),
                np.random.uniform(0.3, 0.5), np.random.uniform(150, 250), np.random.uniform(0.4, 0.6),
                np.random.uniform(0.2, 0.4), np.random.uniform(2, 4), np.random.uniform(15, 30),
                np.random.uniform(5, 10), np.random.uniform(5, 7), np.random.uniform(5, 7),
                np.random.uniform(0.3, 0.5), np.random.uniform(150, 250)
            ]
        elif state == 3:  # Low Energy
            features = [
                np.random.uniform(1, 3), np.random.uniform(50, 150), np.random.uniform(1, 3),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.1, 0.3), np.random.uniform(0.2, 0.4),
                np.random.uniform(0.3, 0.5), np.random.uniform(0.5, 0.7), np.random.uniform(0.6, 0.8),
                np.random.uniform(0.1, 0.2), np.random.uniform(80, 180), np.random.uniform(0.3, 0.5),
                np.random.uniform(0.2, 0.4), np.random.uniform(1, 3), np.random.uniform(5, 15),
                np.random.uniform(2, 5), np.random.uniform(3, 5), np.random.uniform(3, 5),
                np.random.uniform(0, 0.2), np.random.uniform(60, 150)
            ]
        elif state == 4:  # Neutral
            features = [
                np.random.uniform(4, 8), np.random.uniform(150, 300), np.random.uniform(3, 5),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.4), np.random.uniform(0.3, 0.5),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.3, 0.5),
                np.random.uniform(0.2, 0.4), np.random.uniform(100, 200), np.random.uniform(0.2, 0.4),
                np.random.uniform(0.3, 0.5), np.random.uniform(1, 3), np.random.uniform(10, 20),
                np.random.uniform(3, 7), np.random.uniform(5, 7), np.random.uniform(5, 7),
                np.random.uniform(0.2, 0.4), np.random.uniform(100, 180)
            ]
        else:  # Distressed
            features = [
                np.random.uniform(15, 25), np.random.uniform(100, 300), np.random.uniform(7, 12),
                np.random.uniform(0.7, 0.9), np.random.uniform(0.05, 0.15), np.random.uniform(0.4, 0.6),
                np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7),
                np.random.uniform(0.2, 0.4), np.random.uniform(250, 450), np.random.uniform(0.6, 0.9),
                np.random.uniform(0, 0.2), np.random.uniform(3, 6), np.random.uniform(30, 50),
                np.random.uniform(10, 20), np.random.uniform(5, 7), np.random.uniform(5, 7),
                np.random.uniform(0.7, 0.9), np.random.uniform(300, 450)
            ]
        
        X.append(features)
        y.append(state)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Generate training data
X, y = generate_emotional_data(2000)
y_categorical = tf.keras.utils.to_categorical(y, 6)

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 emotion classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ],
    verbose=1
)

# Evaluate
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
print(f"\nTraining accuracy: {train_acc*100:.2f}%")
print(f"Validation accuracy: {val_acc*100:.2f}%")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save model
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\nModel saved: emotion_model.tflite ({len(tflite_model)/1024:.2f} KB)")
