# Emotion Analyzer Models

Two TFLite models for emotion classification:
1. **Behavioral Model** - Analyzes 20 behavioral features
2. **Text Model** - Analyzes text input

## Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train behavioral model
python create_model.py

# Train text model (requires go_emotions_dataset.csv)
python create_text_model.py
```

## Testing

```bash
# Test text analyzer
python test_text_analyzer.py
```

## Models

### Behavioral Model
- **Input**: 20 behavioral features (float32)
- **Output**: 6 emotions (Calm, Restless, Stressed, Low Energy, Neutral, Distressed)
- **File**: emotion_model.tflite (~30 KB)

### Text Model
- **Input**: Text string (max 50 tokens)
- **Output**: ALL 28 emotions from GoEmotions dataset
- **Files**: emotion_text_model.tflite (~300 KB), tokenizer.json

## Emotions

**Behavioral**: Calm, Restless, Stressed, Low Energy, Neutral, Distressed
**Text**: Neutral, Joy, Sadness, Anger, Fear, Surprise
