# Emotion Analyzer Model

TensorFlow Lite model for classifying user emotional states based on behavioral patterns.

## Model Details

- **Input**: 20 behavioral features (float32)
- **Output**: 6 emotion classes
- **Size**: ~30 KB (optimized for mobile)
- **Format**: TFLite with float16 quantization

## Emotion Classes

| Index | Emotion | Description |
|-------|---------|-------------|
| 0 | Calm | Low activity, positive sentiment, regular patterns |
| 1 | Restless | High frequency, short sessions, erratic behavior |
| 2 | Stressed | Long sessions, late night activity, negative sentiment |
| 3 | Low Energy | Minimal activity, low interaction speed |
| 4 | Neutral | Balanced activity and sentiment |
| 5 | Distressed | Very high frequency, extreme negative sentiment |

## Input Features (20 total)

1. avg_open_count
2. avg_session_time
3. avg_interaction_speed
4. late_night_ratio
5. morning_ratio
6. evening_ratio
7. weekend_ratio
8. weekday_ratio
9. short_session_ratio
10. long_session_ratio
11. session_variance
12. negative_ratio
13. positive_ratio
14. note_frequency
15. open_count_variance
16. speed_variance
17. data_points
18. days_covered
19. high_frequency_ratio
20. erratic_behavior_score

## Usage

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Generate model
python create_model.py
```

### Validation

```bash
# Test model accuracy
python validate_model.py
```

### Adding New Emotions

1. Edit `create_model.py`
2. Update `np.random.randint(0, 6)` to `np.random.randint(0, N)` where N = total emotions
3. Add new emotion feature patterns in `generate_emotional_data()`
4. Update `to_categorical(y, 6)` to `to_categorical(y, N)`
5. Change final Dense layer: `Dense(6, ...)` to `Dense(N, ...)`
6. Retrain: `python create_model.py`

## Model Performance

- **Accuracy**: 100% on synthetic data
- **Training samples**: 2000
- **Validation split**: 20%

## Files

- `create_model.py` - Model training script
- `validate_model.py` - Model validation script
- `emotion_model.tflite` - Generated TFLite model
- `requirements.txt` - Python dependencies
