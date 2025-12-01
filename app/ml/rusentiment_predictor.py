import joblib
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.data.dataset_loader import RuSentimentLoader

logger = logging.getLogger(__name__)


def _create_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words=None
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=2000,
            C=0.5,
            solver='saga',
            multi_class='multinomial'
        ))
    ])


def _get_fallback_data() -> Tuple[List[str], List[int]]:
    logger.info("Using fallback training data")

    texts = []
    labels = []

    positive_examples = [
        "отличный товар рекомендую всем",
        "прекрасное качество очень доволен",
        "супер купил не пожалел",
        "восхитительно просто шикарно",
        "люблю этот продукт лучший",
        "великолепный продукт советую",
        "превосходное качество рад покупке",
        "идеально работает как надо",
        "лучшее что я покупал",
        "высокое качество доволен на все сто",
        "отличная работа молодец",
        "суперский товар всем советую",
        "замечательно все устроило",
        "прекрасно справляется с задачами",
        "очень хороший продукт"
    ]

    negative_examples = [
        "плохой товар ужасное качество",
        "не рекомендую разочарован покупкой",
        "кошмар а не продукт",
        "очень плохо не покупайте",
        "гадкий отвратительный товар",
        "ужасный сервис неудовлетворен",
        "отвратительно работает",
        "не стоит денег выброшенные средства",
        "худший товар что я покупал",
        "полный разочарование",
        "низкое качество материалов",
        "брак не работает",
        "обман не соответствует описанию",
        "мусор а не товар",
        "не функционирует сломался сразу"
    ]

    other_examples = [
        "нормальный товар ничего особенного",
        "сойдет можно пользоваться",
        "неплохо но есть недостатки",
        "среднего качества как все",
        "может быть подойдет не уверен",
        "обычный продукт без изысков",
        "нейтральный отзыв ничего особого",
        "такой себе ничего примечательного",
        "стандартный товар как у всех",
        "не плохо но и не хорошо",
        "средненько обыкновенно",
        "приемлемо можно использовать",
        "не выделяется среди других",
        "обычно нормально стандартно",
        "посредственный продукт"
    ]

    texts = positive_examples + negative_examples + other_examples
    labels = [1] * len(positive_examples) + [0] * len(negative_examples) + [2] * len(other_examples)

    return texts, labels


class RuSentimentPredictor:
    def __init__(self, model_path: Optional[str] = None, data_path: Optional[str] = None):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.classes = ['negative', 'positive', 'other']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.data_loader = RuSentimentLoader(data_path)

        logger.info("RuSentimentPredictor initialized")

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.lower().strip()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^а-яё0-9\s\.\,\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'не\s+', 'не_', text)
        text = re.sub(r'ни\s+', 'ни_', text)
        text = re.sub(r'без\s+', 'без_', text)
        text = re.sub(r'\!+', '!', text)
        text = re.sub(r'\?+', '?', text)

        return text

    def predict(self, text: str) -> str:
        if self.model is None:
            self._load_or_create_model()

        processed = self.preprocess_text(text)
        prediction_idx = self.model.predict([processed])[0]
        return self.idx_to_class.get(prediction_idx, 'other')

    def predict_with_confidence(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None:
            self._load_or_create_model()

        processed = self.preprocess_text(text)

        try:
            probabilities = self.model.predict_proba([processed])[0]
        except (AttributeError, IndexError) as e:
            logger.warning(f"Error getting probabilities: {e}")
            prediction_idx = self.model.predict([processed])[0]
            probabilities = np.zeros(len(self.classes))
            probabilities[prediction_idx] = 1.0

        predicted_class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_idx])
        predicted_class = self.idx_to_class.get(predicted_class_idx, 'other')

        class_probabilities = {
            self.classes[i]: float(probabilities[i])
            for i in range(len(self.classes))
        }

        return predicted_class, confidence, class_probabilities

    def _load_or_create_model(self):
        if self.model_path and Path(self.model_path).exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

        logger.info("Training new model on RuSentiment data...")
        self._train_model()

    def _train_model(self):
        if self.data_path:
            texts, labels = self._load_training_data()
        else:
            texts, labels = _get_fallback_data()

        if len(texts) < 100:
            logger.warning(f"Very small dataset: {len(texts)} samples. Using fallback.")
            texts, labels = _get_fallback_data()

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.model = _create_pipeline()
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        logger.info(f"Model trained. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes))

        if self.model_path:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

    def _load_training_data(self) -> Tuple[List[str], List[int]]:
        logger.info(f"Loading training data from {self.data_path}")

        try:
            self.data_loader.load(self.data_path)
            text_col, label_col = self.data_loader.explore_dataset()

            if text_col and label_col:
                texts, labels = self.data_loader.prepare_for_training(
                    text_col, label_col, max_samples_per_class=2000
                )

                if len(texts) > 100:
                    return texts, labels
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")

        return _get_fallback_data()

    def evaluate(self, test_texts: List[str], test_labels: List[int]):
        if self.model is None:
            self._load_or_create_model()

        predictions = []
        confidences = []

        for text in test_texts:
            pred, conf, _ = self.predict_with_confidence(text)
            predictions.append(self.class_to_idx.get(pred, 2))
            confidences.append(conf)

        accuracy = np.mean([1 if p == l else 0 for p, l in zip(predictions, test_labels)])
        avg_confidence = np.mean(confidences)

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Average confidence: {avg_confidence:.3f}")

        return accuracy, avg_confidence


def test_predictor():
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА RUSENTIMENT")
    print("=" * 70)

    predictor = RuSentimentPredictor(
        model_path="models/rusentiment_model.joblib",
        data_path="data/rusentiment.csv"
    )

    test_texts = [
        "Отличный товар! Очень рекомендую!",
        "Ужасное качество, не покупайте",
        "Нормальный продукт, но дорогой",
        "Супер, мне очень понравилось!",
        "Кошмар, полный разочарование",
        "Ничего особенного, обычный товар"
    ]

    for text in test_texts:
        print(f"\nТекст: {text}")
        pred, conf, probs = predictor.predict_with_confidence(text)
        print(f"  Предсказание: {pred}")
        print(f"  Уверенность: {conf:.3f}")
        print(f"  Вероятности: {probs}")

    print("\n" + "=" * 70)
    print("Тестирование завершено")
    print("=" * 70)

    return predictor


if __name__ == "__main__":
    test_predictor()