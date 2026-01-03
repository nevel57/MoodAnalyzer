<p align="center">🧠 RuSentiment Hybrid Analytics</p>

<p align="center">
<a href="#">
<img src="[https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Python-3.9%2B-3776ab%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite)" alt="Python">
</a>
<a href="#">
<img src="[https://img.shields.io/badge/FastAPI-0.100+-05998b?style=for-the-badge&logo=fastapi&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/FastAPI-0.100%2B-05998b%3Fstyle%3Dfor-the-badge%26logo%3Dfastapi%26logoColor%3Dwhite)" alt="FastAPI">
</a>
<a href="#">
<img src="[https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Machine%2520Learning-Scikit--Learn-f7931e%3Fstyle%3Dfor-the-badge%26logo%3Dscikit-learn%26logoColor%3Dwhite)" alt="ML">
</a>
</p>

<p align="center">
<b>Высокопроизводительный сервис анализа тональности для русского языка</b>




<i>Интеллектуальный баланс между скоростью классического ML и глубиной нейросетей.</i>
</p>

---

## ⚡ Визуализация архитектуры

Система использует **Smart Routing**, чтобы экономить ваши ресурсы без потери качества.

1. **Вход:** Текст поступает в систему.
2. **Маршрутизатор:** Оценивает длину и сложность (наличие сарказма, отрицаний).
3. **Выбор пути:**
* **Fast Path:** Обработка за **<5 мс** через оптимизированный LogReg.
* **Accurate Path:** Глубокий анализ через **BERT-контур** для сложных случаев.



---

## 🔥 Ключевые возможности

|  | Функция | Описание |
| --- | --- | --- |
| 🚀 | **Ultra-Fast** | Обработка до 500 RPS на обычном CPU. |
| 🤖 | **Multi-Model** | Переключение между 4-мя специализированными профилями. |
| 📊 | **Confidence Score** | Не просто результат, а точная степень уверенности модели. |
| 🧪 | **Live Dashboard** | Встроенный UI для мгновенного тестирования гипотез. |

---

## 📂 Доступные профили моделей

```mermaid
graph LR
    A[Клиент] --> B{Выбор модели}
    B --> C[Fresh: Стандарт]
    B --> D[Combined: Макс. данных]
    B --> E[Quality: Отзывы]
    B --> F[Optimized: RAM-сэйвер]

```

---

## 🚀 Быстрый старт

### 🛠 Установка окружения

```bash
# Клонируем и заходим
git clone https://github.com/your-username/rusentiment-hybrid.git && cd rusentiment-hybrid

# Ставим всё необходимое
pip install -r requirements.txt

```

### 🛰 Запуск ядра

```bash
python app/run_rusentiment.py

```

> **Info:** Документация Swagger автоматически доступна по адресу: `http://localhost:8000/docs`

---

## 📈 Пример аналитики

API возвращает детальный JSON с картой вероятностей:

```json
{
  "sentiment": "positive",
  "confidence": 0.942,
  "probabilities": {
    "positive": 0.94,
    "negative": 0.02,
    "neutral": 0.04
  },
  "routing": "fast_path_optimized"
}

```

---

## 🤝 Контакты

Если вам нравится проект, поставьте ему ⭐. Это помогает развитию!

---
