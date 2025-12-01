import sys
import os
sys.path.append('app')

from app.ml.rusentiment_predictor import RuSentimentPredictor

print("=" * 60)
print("ПРОВЕРКА ОБУЧЕНИЯ МОДЕЛИ")
print("=" * 60)

print("\n1. Обучаю новую модель...")
predictor = RuSentimentPredictor(
    model_path="models/check_training.joblib",
    data_path="data/rusentiment.csv"
)

predictor._train_model()

print("\n2. Тестирую...")
test_texts = [
    ("Отличный товар!", "positive"),
    ("Ужасное качество", "negative"),
    ("Нормально, ничего особенного", "other"),
    ("Супер, мне понравилось!", "positive"),
    ("Плохой продукт", "negative"),
]

for text, expected in test_texts:
    pred, conf, probs = predictor.predict_with_confidence(text)
    status = "✅" if pred == expected else "❌"
    print(f"\n{status} Текст: {text}")
    print(f"   Ожидалось: {expected}, Получено: {pred}")
    print(f"   Уверенность: {conf:.3f}")
    print(f"   Вероятности: {probs}")

print("\n" + "=" * 60)
print("Проверка завершена")
print("=" * 60)