import re
import time
from typing import Dict, Any
import logging
import numpy as np

from app.ml.rusentiment_predictor import RuSentimentPredictor
from app.router.request_router import RequestRouter

logger = logging.getLogger(__name__)


class AccuratePredictorStub:
    def __init__(self):
        self.model_type = "bert_stub"
        logger.info("AccuratePredictorStub initialized")

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def predict_with_confidence(self, text: str) -> tuple:
        processed = self.preprocess_text(text)

        temp_predictor = RuSentimentPredictor()
        temp_predictor._load_or_create_model()

        pred, conf, probs = temp_predictor.predict_with_confidence(text)

        boosted_conf = min(conf * 1.2, 0.95)

        for key in probs:
            if key == pred:
                probs[key] = boosted_conf
            else:
                probs[key] = (1 - boosted_conf) / (len(probs) - 1)

        return pred, boosted_conf, probs


class HybridSentimentServiceRuSentiment:
    def __init__(self):
        self.fast_predictor = RuSentimentPredictor("models/rusentiment_fast.joblib")
        self.accurate_predictor = AccuratePredictorStub()
        self.router = RequestRouter()
        logger.info("HybridSentimentServiceRuSentiment initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return self._error_response("Empty text")

        if len(text) > 1000:
            text = text[:1000]
            logger.warning(f"Text truncated to 1000 characters")

        try:
            start_time = time.time()
            fast_pred, fast_conf, fast_probs = self.fast_predictor.predict_with_confidence(text)
            fast_time = time.time() - start_time

            start_time = time.time()
            use_fast, routing_info = self.router.route_request(text, (fast_pred, fast_conf, fast_probs))
            routing_time = time.time() - start_time

            if use_fast:
                final_pred, final_conf, final_probs = fast_pred, fast_conf, fast_probs
                model_used = "rusentiment_fast"
                model_time = fast_time
            else:
                start_time = time.time()
                final_pred, final_conf, final_probs = self.accurate_predictor.predict_with_confidence(text)
                model_time = time.time() - start_time
                model_used = "accurate_stub"

            return {
                'success': True,
                'text': text,
                'sentiment': final_pred,
                'confidence': round(final_conf, 3),
                'probabilities': final_probs,
                'model_used': model_used,
                'routing_decision': {
                    'use_fast': use_fast,
                    'reason': routing_info['reason'],
                    'fast_confidence': routing_info.get('fast_confidence'),
                    'text_complexity': routing_info.get('text_complexity'),
                },
                'performance': {
                    'total_time': round(fast_time + routing_time + model_time, 4),
                    'fast_model_time': round(fast_time, 4),
                    'routing_time': round(routing_time, 4),
                    'final_model_time': round(model_time, 4),
                },
                'text_info': {
                    'length': len(text),
                    'word_count': len(text.split()),
                }
            }

        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}")
            return self._error_response(f"Analysis error: {str(e)}")

    def batch_analyze(self, texts: list[str]) -> list[Dict[str, Any]]:
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)

        fast_count = sum(1 for r in results if r.get('success') and r['routing_decision']['use_fast'])
        accurate_count = len([r for r in results if r.get('success')]) - fast_count

        logger.info(f"Batch analysis: {fast_count} by FAST, {accurate_count} by ACCURATE")

        return results

    @staticmethod
    def _error_response(message: str) -> Dict[str, Any]:
        return {
            'success': False,
            'error': message,
            'text': None,
            'sentiment': None,
            'confidence': 0.0,
            'probabilities': {},
            'model_used': None,
            'routing_decision': None,
            'performance': None,
            'text_info': None
        }


def test_hybrid_service():
    service = HybridSentimentServiceRuSentiment()

    test_cases = [
        ("Отличный товар! Очень рекомендую!", "simple positive"),
        ("Ужасное качество, не покупайте", "simple negative"),
        ("Очень длинный текст " + "очень " * 30, "long text"),
        ("Сарказм? Да, конечно, отлично...", "sarcasm"),
        ("Не без оснований", "double negation"),
        ("Нормально, но могло быть лучше", "neutral with contrast"),
    ]

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ГИБРИДНОГО СЕРВИСА (RuSentiment)")
    print("=" * 70)

    for text, description in test_cases:
        print(f"\n📝 [{description}]")
        print(f"   Текст: {text[:60]}..." if len(text) > 60 else f"   Текст: {text}")

        result = service.analyze(text)

        if result['success']:
            print(f"   ✅ Результат: {result['sentiment']}")
            print(f"   📊 Уверенность: {result['confidence']:.2f}")
            print(f"   🤖 Модель: {result['model_used'].upper()}")
            print(f"   🎯 Решение: {result['routing_decision']['reason']}")
            print(f"   ⏱️  Время: {result['performance']['total_time']:.3f}s")
        else:
            print(f"   ❌ Ошибка: {result['error']}")

    print("\n" + "=" * 70)
    print("Тестирование завершено")
    print("=" * 70)

    return service


if __name__ == "__main__":
    test_hybrid_service()