import re
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    use_fast_model: bool
    reason: str
    confidence: Optional[float] = None
    text_complexity: Optional[float] = None


def _default_config() -> Dict[str, Any]:
    return {
        'min_fast_confidence': 0.7,
        'max_text_length_fast': 200,
        'complexity_word_count_threshold': 20,
        'complexity_special_chars_threshold': 0.3,
        'accurate_patterns': [
            'сарказм', 'ирония', 'конечно', 'ещё бы',
            'наверное', 'возможно', 'может быть', 'скорее всего',
            'с одной стороны', 'с другой стороны',
            'если бы', 'хотелось бы', 'желательно',
            'по сравнению', 'в отличие от'
        ],
        'cache_enabled': True,
        'timeout_fast_ms': 100,

    }


class RequestRouter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or _default_config()
        logger.info(f"RequestRouter initialized with config: {self.config}")

    def analyze_text_complexity(self, text: str) -> float:
        if not text:
            return 0.0

        score = 0.0
        text_len = len(text)

        length_factor = min(text_len / self.config['max_text_length_fast'], 1.0)
        score += length_factor * 0.3

        words = text.split()
        word_count = len(words)
        if word_count > self.config['complexity_word_count_threshold']:
            score += 0.2

        special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
        special_ratio = special_chars / text_len if text_len > 0 else 0
        if special_ratio > self.config['complexity_special_chars_threshold']:
            score += 0.2

        complex_patterns = [
            'но', 'однако', 'хотя', 'несмотря на', 'в то время как',
            'тем не менее', 'впрочем', 'следовательно', 'таким образом'
        ]
        for pattern in complex_patterns:
            if pattern in text.lower():
                score += 0.1
                break

        if '?' in text and '!' in text:
            score += 0.2

        negation_patterns = [r'не\s+не', r'ни\s+ни', r'без\s+не']
        for pattern in negation_patterns:
            if re.search(pattern, text.lower()):
                score += 0.1
                break

        return min(max(score, 0.0), 1.0)

    def should_use_accurate_model(self, text: str, fast_confidence: Optional[float] = None) -> RoutingDecision:
        if not text or len(text.strip()) == 0:
            return RoutingDecision(
                use_fast_model=True,
                reason="Empty text",
                confidence=1.0,
                text_complexity=0.0
            )

        text_lower = text.lower()

        for pattern in self.config['accurate_patterns']:
            if pattern in text_lower:
                return RoutingDecision(
                    use_fast_model=False,
                    reason=f"Contains pattern: '{pattern}'",
                    confidence=fast_confidence,
                    text_complexity=1.0
                )
        complexity = self.analyze_text_complexity(text)

        if len(text) > self.config['max_text_length_fast']:
            return RoutingDecision(
                use_fast_model=False,
                reason=f"Text too long ({len(text)} chars)",
                confidence=fast_confidence,
                text_complexity=complexity
            )

        if fast_confidence is not None:
            if fast_confidence < self.config['min_fast_confidence']:
                return RoutingDecision(
                    use_fast_model=False,
                    reason=f"Low fast model confidence: {fast_confidence:.2f}",
                    confidence=fast_confidence,
                    text_complexity=complexity
                )

        if complexity > 0.7:  # Высокая сложность
            return RoutingDecision(
                use_fast_model=False,
                reason=f"High text complexity: {complexity:.2f}",
                confidence=fast_confidence,
                text_complexity=complexity
            )

        return RoutingDecision(
            use_fast_model=True,
            reason="Simple text with sufficient confidence",
            confidence=fast_confidence,
            text_complexity=complexity
        )

    def route_request(
            self,
            text: str,
            fast_prediction: Optional[Tuple[str, float, Dict[str, float]]] = None
    ) -> Tuple[bool, Dict[str, Any]]:

        fast_confidence = None
        if fast_prediction:
            _, fast_confidence, _ = fast_prediction

        decision = self.should_use_accurate_model(text, fast_confidence)

        routing_info = {
            'use_fast_model': decision.use_fast_model,
            'reason': decision.reason,
            'fast_confidence': decision.confidence,
            'text_complexity': decision.text_complexity,
            'text_length': len(text),
            'word_count': len(text.split()),
            'decision_time': 'immediate',
        }

        logger.debug(f"Routing decision for text (first 50 chars): '{text[:50]}...'")
        logger.info(f"Routing: {'FAST' if decision.use_fast_model else 'ACCURATE'} - {decision.reason}")

        return decision.use_fast_model, routing_info

    def batch_route(self, texts: list[str]) -> list[Tuple[bool, Dict[str, Any]]]:
        decisions = []
        for text in texts:
            decision = self.route_request(text)
            decisions.append(decision)

        fast_count = sum(1 for use_fast, _ in decisions if use_fast)
        accurate_count = len(decisions) - fast_count

        logger.info(f"Batch routing: {fast_count} to FAST, {accurate_count} to ACCURATE")

        return decisions

def create_default_router(self) -> RequestRouter:
    return RequestRouter()

def demonstrate_routing():
    router = RequestRouter()
    examples = [
        # для FAST
        ("Отличный товар!", "simple positive"),
        ("Плохое качество.", "simple negative"),

        # для ACCURATE
        ("Очень длинный текст " + "очень " * 50, "very long text"),

        # сложн предлож в ACCURATE
        ("Это не плохо, а очень даже хорошо, но есть нюансы...", "complex with 'но'"),
        ("Возможно, это хороший товар, но я не уверен.", "uncertainty"),

        # сарказм в ACCURATE
        ("Сарказм? Да, конечно, отличное качество...", "sarcasm"),

        # смешанные эмоции в ACCURATE
        ("Неплохо, но могло быть и лучше! А почему так дорого?", "mixed emotions"),

        # короткий но сложный  в ACCURATE
        ("Не без оснований", "double negation"),
    ]

    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ МАРШРУТИЗАТОРА")
    print("=" * 60)

    for i, (text, description) in enumerate(examples, 1):
        use_fast, info = router.route_request(text)
        model = "FAST" if use_fast else "ACCURATE"

        print(f"\n{i}. [{description}]")
        print(f"   Текст: {text[:60]}..." if len(text) > 60 else f"   Текст: {text}")
        print(f"   Решение: {model}")
        print(f"   Причина: {info['reason']}")
        print(f"   Сложность: {info['text_complexity']:.2f}")
        print(f"   Длина: {info['text_length']} chars, {info['word_count']} words")

    print("\n" + "=" * 60)
    print("Демонстрация завершена")
    print("=" * 60)

    return router

if __name__ == "__main__":
    demonstrate_routing()