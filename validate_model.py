import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

def generate_emotional_data(n_samples=2000):
    X, y = [], []
    for _ in range(n_samples):
        state = np.random.randint(0, 6)
        if state == 0:  # Calm
            features = [np.random.uniform(3, 6), np.random.uniform(120, 300), np.random.uniform(2, 4),
                       np.random.uniform(0, 0.2), np.random.uniform(0.3, 0.5), np.random.uniform(0.2, 0.4),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.1, 0.3),
                       np.random.uniform(0.2, 0.4), np.random.uniform(50, 150), np.random.uniform(0, 0.2),
                       np.random.uniform(0.5, 0.8), np.random.uniform(1, 3), np.random.uniform(5, 15),
                       np.random.uniform(2, 5), np.random.uniform(5, 7), np.random.uniform(5, 7),
                       np.random.uniform(0, 0.2), np.random.uniform(40, 120)]
        elif state == 1:  # Restless
            features = [np.random.uniform(10, 15), np.random.uniform(30, 120), np.random.uniform(6, 10),
                       np.random.uniform(0.3, 0.5), np.random.uniform(0.1, 0.3), np.random.uniform(0.3, 0.5),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8),
                       np.random.uniform(0.1, 0.2), np.random.uniform(200, 400), np.random.uniform(0.4, 0.7),
                       np.random.uniform(0.1, 0.3), np.random.uniform(2, 5), np.random.uniform(20, 40),
                       np.random.uniform(8, 15), np.random.uniform(5, 7), np.random.uniform(5, 7),
                       np.random.uniform(0.5, 0.7), np.random.uniform(200, 350)]
        elif state == 2:  # Stressed
            features = [np.random.uniform(8, 12), np.random.uniform(200, 400), np.random.uniform(4, 7),
                       np.random.uniform(0.6, 0.8), np.random.uniform(0.1, 0.2), np.random.uniform(0.4, 0.6),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.3, 0.5),
                       np.random.uniform(0.3, 0.5), np.random.uniform(150, 250), np.random.uniform(0.4, 0.6),
                       np.random.uniform(0.2, 0.4), np.random.uniform(2, 4), np.random.uniform(15, 30),
                       np.random.uniform(5, 10), np.random.uniform(5, 7), np.random.uniform(5, 7),
                       np.random.uniform(0.3, 0.5), np.random.uniform(150, 250)]
        elif state == 3:  # Low Energy
            features = [np.random.uniform(1, 3), np.random.uniform(50, 150), np.random.uniform(1, 3),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.1, 0.3), np.random.uniform(0.2, 0.4),
                       np.random.uniform(0.3, 0.5), np.random.uniform(0.5, 0.7), np.random.uniform(0.6, 0.8),
                       np.random.uniform(0.1, 0.2), np.random.uniform(80, 180), np.random.uniform(0.3, 0.5),
                       np.random.uniform(0.2, 0.4), np.random.uniform(1, 3), np.random.uniform(5, 15),
                       np.random.uniform(2, 5), np.random.uniform(3, 5), np.random.uniform(3, 5),
                       np.random.uniform(0, 0.2), np.random.uniform(60, 150)]
        elif state == 4:  # Neutral
            features = [np.random.uniform(4, 8), np.random.uniform(150, 300), np.random.uniform(3, 5),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.2, 0.4), np.random.uniform(0.3, 0.5),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.3, 0.5),
                       np.random.uniform(0.2, 0.4), np.random.uniform(100, 200), np.random.uniform(0.2, 0.4),
                       np.random.uniform(0.3, 0.5), np.random.uniform(1, 3), np.random.uniform(10, 20),
                       np.random.uniform(3, 7), np.random.uniform(5, 7), np.random.uniform(5, 7),
                       np.random.uniform(0.2, 0.4), np.random.uniform(100, 180)]
        else:  # Distressed
            features = [np.random.uniform(15, 25), np.random.uniform(100, 300), np.random.uniform(7, 12),
                       np.random.uniform(0.7, 0.9), np.random.uniform(0.05, 0.15), np.random.uniform(0.4, 0.6),
                       np.random.uniform(0.2, 0.4), np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7),
                       np.random.uniform(0.2, 0.4), np.random.uniform(250, 450), np.random.uniform(0.6, 0.9),
                       np.random.uniform(0, 0.2), np.random.uniform(3, 6), np.random.uniform(30, 50),
                       np.random.uniform(10, 20), np.random.uniform(5, 7), np.random.uniform(5, 7),
                       np.random.uniform(0.7, 0.9), np.random.uniform(300, 450)]
        X.append(features)
        y.append(state)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

print("=" * 60)
print("EMOTION MODEL VALIDATION REPORT")
print("=" * 60)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='emotion_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Generate test data
X, y = generate_emotional_data(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTest samples: {len(X_test)}")
print(f"Class distribution: {np.bincount(y_test)}")

# Run predictions
predictions = []
for sample in X_test:
    interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predictions.append(np.argmax(output))

predictions = np.array(predictions)

# Calculate metrics
accuracy = np.mean(predictions == y_test) * 100
print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy:.2f}%")
print(f"{'='*60}")

# Class names
class_names = ['Calm', 'Restless', 'Stressed', 'Low Energy', 'Neutral', 'Distressed']

# Classification report
print("\nDETAILED CLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(y_test, predictions, target_names=class_names, digits=3))

# Confusion matrix
print("\nCONFUSION MATRIX:")
print("-" * 60)
cm = confusion_matrix(y_test, predictions)
print("Predicted ->")
print(f"{'Actual':<12}", end="")
for name in class_names:
    print(f"{name[:8]:<10}", end="")
print()
for i, name in enumerate(class_names):
    print(f"{name:<12}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i][j]:<10}", end="")
    print()

# Per-class accuracy
print("\nPER-CLASS ACCURACY:")
print("-" * 60)
for i, name in enumerate(class_names):
    class_acc = cm[i][i] / np.sum(cm[i]) * 100 if np.sum(cm[i]) > 0 else 0
    print(f"{name:<15}: {class_acc:>6.2f}%")

# Model info
import os
model_size = os.path.getsize('emotion_model.tflite') / 1024
print(f"\n{'='*60}")
print(f"MODEL INFO:")
print(f"  Size: {model_size:.2f} KB")
print(f"  Input: 20 features (float32)")
print(f"  Output: 6 classes")
print(f"  Format: TFLite (optimized for mobile)")
print(f"{'='*60}")
