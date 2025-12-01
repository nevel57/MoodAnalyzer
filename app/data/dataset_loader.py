import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def save_processed_data(output_path: str, texts: List[str], labels: List[int]):
    df_processed = pd.DataFrame({
        'text': texts,
        'label': labels,
        'label_name': ['negative' if l == 0 else 'positive' if l == 1 else 'other' for l in labels]
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Data saved to {output_path}")

    print(f"\nСохранено: {len(df_processed)} примеров")
    print("Распределение:")
    print(df_processed['label_name'].value_counts())


class RuSentimentLoader:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.df = None

    def load(self, data_path: Optional[str] = None) -> pd.DataFrame:
        path = data_path or self.data_path
        if not path:
            raise ValueError("No data path provided")

        logger.info(f"Loading dataset from {path}")

        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            try:
                self.df = pd.read_csv(path, delimiter='\t')
            except:
                self.df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')

        logger.info(f"Loaded {len(self.df)} rows")
        print("Колонки:", self.df.columns.tolist())
        print("\nПервые 5 строк:")
        print(self.df.head())

        return self.df

    def explore_dataset(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        print("=" * 60)
        print("АНАЛИЗ ДАТАСЕТА")
        print("=" * 60)

        print(f"Всего записей: {len(self.df)}")
        print(f"Колонки: {self.df.columns.tolist()}")

        text_columns = ['text', 'comment', 'sentence', 'tweet', 'content']
        text_column = None
        for col in text_columns:
            if col in self.df.columns:
                text_column = col
                break

        if not text_column:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    text_column = col
                    break

        print(f"Колонка с текстом: {text_column}")

        label_columns = ['label', 'sentiment', 'class', 'category', 'sentiment_label']
        label_column = None
        for col in label_columns:
            if col in self.df.columns:
                label_column = col
                break

        print(f"Колонка с лейблами: {label_column}")

        if label_column:
            print("\nРаспределение классов:")
            label_counts = self.df[label_column].value_counts()
            for label, count in label_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")

        if text_column:
            text_lengths = self.df[text_column].astype(str).apply(len)
            print(f"\nСтатистика длины:")
            print(f"  Мин: {text_lengths.min()}")
            print(f"  Макс: {text_lengths.max()}")
            print(f"  Среднее: {text_lengths.mean():.1f}")

            print(f"\nПримеры текстов:")
            for i in range(min(3, len(self.df))):
                text = str(self.df.iloc[i][text_column])
                label = self.df.iloc[i][label_column] if label_column else 'N/A'
                print(f"  [{label}] {text[:80]}...")

        print("=" * 60)

        return text_column, label_column

    def prepare_for_training(
            self,
            text_column: str = 'text',
            label_column: str = 'label',
            max_samples_per_class: int = 5000
    ) -> Tuple[List[str], List[int]]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        logger.info("Preparing data for 3-class training...")

        label_mapping = {
            'negative': 0, 'neg': 0, '0': 0,
            'positive': 1, 'pos': 1, '1': 1,
            'neutral': 2, 'neu': 2, '2': 2,
            'speech': 2, 'spe': 2,
            'skip': 2
        }

        texts = []
        labels = []
        class_counts = {0: 0, 1: 0, 2: 0}

        for _, row in self.df.iterrows():
            text = str(row[text_column]).strip()
            original_label = str(row[label_column]).strip().lower()

            if not text or len(text) < 2:
                continue

            mapped_label = None
            for key, value in label_mapping.items():
                if key in original_label:
                    mapped_label = value
                    break

            if mapped_label is None:
                continue

            if class_counts[mapped_label] >= max_samples_per_class:
                continue

            texts.append(text)
            labels.append(mapped_label)
            class_counts[mapped_label] += 1

            if all(count >= max_samples_per_class for count in class_counts.values()):
                break

        logger.info(f"Prepared {len(texts)} samples")
        logger.info(f"Classes: negative={class_counts[0]}, positive={class_counts[1]}, other={class_counts[2]}")

        return texts, labels


def test_loader():
    print("Testing RuSentiment loader...")

    loader = RuSentimentLoader()

    try:
        df = loader.load("data/rusentiment.csv")
        text_col, label_col = loader.explore_dataset()

        if text_col and label_col:
            texts, labels = loader.prepare_for_training(text_col, label_col, 2000)
            print(f"\nГотово для обучения: {len(texts)} примеров")

            save_processed_data("data/processed_rusentiment.csv", texts, labels)
            return texts, labels
    except Exception as e:
        print(f"Ошибка: {e}")
        print("\nСоздаю тестовые данные...")

        test_texts = [
            "отличный товар рекомендую",
            "плохое качество не покупайте",
            "нормальный продукт сойдет",
            "супер мне понравилось",
            "ужасный сервис разочарован",
            "ничего особенного обычный"
        ]
        test_labels = [1, 0, 2, 1, 0, 2]

        return test_texts, test_labels


if __name__ == "__main__":
    test_loader()