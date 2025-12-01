import sys
import os

sys.path.append('app')

from app.ml.rusentiment_predictor import RuSentimentPredictor


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ НА RUSENTIMENT")
    print("=" * 60)

    # Создаем папку для моделей если нет
    os.makedirs("models", exist_ok=True)

    # Инициализируем и обучаем модель
    predictor = RuSentimentPredictor(
        model_path="models/rusentiment_trained.joblib",
        data_path="data/rusentiment.csv"
    )

    print("\nНачинаю обучение...")
    predictor._load_or_create_model()  # Это запустит обучение

    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print(f"Модель сохранена: models/rusentiment_trained.joblib")
    print("=" * 60)

    # Тестируем модель
    print("\n📊 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ:")

    test_cases = [
        ("Отличный товар! Очень рекомендую!", "positive"),
        ("Ужасное качество, не покупайте", "negative"),
        ("Нормальный продукт, ничего особенного", "other"),
        ("Супер, мне очень понравилось!", "positive"),
        ("Кошмар, полный разочарование", "negative"),
    ]

    for text, expected in test_cases:
        pred, conf, probs = predictor.predict_with_confidence(text)
        status = "✅" if pred == expected else "❌"
        print(f"\n{status} Текст: {text}")
        print(f"   Ожидалось: {expected}, Получено: {pred}")
        print(f"   Уверенность: {conf:.3f}")

    print("\n" + "=" * 60)
    print("Готово! Можете запускать API с обученной моделью.")
    print("=" * 60)


if __name__ == "__main__":
    main()