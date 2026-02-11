# Emotion Analyzer Model

Hybrid TensorFlow Lite models for emotion classification:
1. **Behavioral Model**: Analyzes user activity patterns
2. **Text Model**: Analyzes user text input
3. **Hybrid System**: Combines both for comprehensive analysis

## Models

### Behavioral Model
- **Input**: 20 behavioral features (float32)
- **Output**: 6 emotion classes
- **Size**: ~30 KB
- **File**: emotion_model.tflite

### Text Model
- **Input**: Text sequence (50 tokens)
- **Output**: 6 emotion classes
- **Size**: Variable
- **File**: emotion_text_model.tflite

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

### Training Behavioral Model

```bash
pip install -r requirements.txt
python create_model.py
```

### Training Text Model

```bash
python create_text_model.py
```

### Hybrid Analysis

```python
from hybrid_analyzer import HybridEmotionAnalyzer

analyzer = HybridEmotionAnalyzer()

# Behavioral only
result = analyzer.hybrid_analysis(behavioral_features=[...])

# Text only
result = analyzer.hybrid_analysis(text_sequence=[...])

# Both (60% behavioral, 40% text)
result = analyzer.hybrid_analysis(
    behavioral_features=[...],
    text_sequence=[...],
    behavior_weight=0.6
)
```

### Adding New Emotions

1. Edit `create_model.py`
2. Update `np.random.randint(0, 6)` to `np.random.randint(0, N)` where N = total emotions
3. Add new emotion feature patterns in `generate_emotional_data()`
4. Update `to_categorical(y, 6)` to `to_categorical(y, N)`
5. Change final Dense layer: `Dense(6, ...)` to `Dense(N, ...)`
6. Retrain: `python create_model.py`

## Model Performance

- **Accuracy**: 100% on synthetic behavioral data
- **Training samples**: 2000
- **Validation split**: 20%
- **Model validated**: Yes (see validate_model.py)

## Notes

- Model uses synthetic behavioral patterns optimized for user activity tracking
- GoEmotions dataset (text-based) not directly applicable for behavioral features
- Current model designed for 20 behavioral metrics, not text sentiment

## Files

- `create_model.py` - Behavioral model training
- `create_text_model.py` - Text model training
- `hybrid_analyzer.py` - Hybrid analysis system
- `validate_model.py` - Model validation
- `emotion_model.tflite` - Behavioral model
- `emotion_text_model.tflite` - Text model
- `tokenizer.json` - Text tokenizer
- `requirements.txt` - Dependencies
