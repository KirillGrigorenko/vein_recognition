# Модель распознавания по венам ладони

## Что это за модель?

**Задача:** Биометрическая идентификация человека по фотографии вен ладони.

**Как работает:**
- Модель смотрит на снимок ладони и превращает его в вектор из 512 чисел (эмбеддинг)
- Потом сравнивает этот вектор с эмбеддингами из базы (галереи)
- Находит ближайший по косинусному расстоянию → это и есть предсказанный человек

**Архитектура:** ResNet18 (предобученный на ImageNet, адаптированный под grayscale)
- Вход: grayscale изображение 224×224 (1 канал)
- Выход: L2-нормализованный вектор 512 измерений
- Метрика сравнения: косинусное сходство (чем ближе к 1 — тем похожее)

**Обучение:**
- 834 человека, по 10 фото каждый (8340 изображений всего)
- 8 фото на обучение / 2 на тест (на каждого человека)
- Loss: CrossEntropy + метрическое обучение (Triplet Loss)
- Лучшая точность на валидации: **88%** (Top-1)
- Обучалось ~80 эпох

---

## Файлы которые ты получил

```
vein_recognition/
├── src/                        # Весь код
│   ├── model_triplet.py        # Архитектура модели
│   ├── config.py               # Гиперпараметры
│   ├── vein_data_loader.py     # Загрузка датасета
│   ├── train.py                # Обучение
│   ├── test_accuracy.py        # Проверка точности на датасете
│   ├── identify_person.py      # Идентификация по одному фото
│   ├── find_min_samples.py     # Мин. кол-во фото для одного человека
│   └── find_min_samples_mass.py# То же, но массово
├── results/
│   └── best/
│       └── huita.pt            # <-- ВЕСА МОДЕЛИ (используй этот файл)
├── data/
│   └── raw/                    # Датасет: папки 1-834, в каждой 10 .bmp фото
│       └── 1/, 2/, ... 834/
│           └── 1_1.bmp ... 1_10.bmp
└── requirements.txt
```

**Файл весов: `results/best/huita.pt`**
- Формат: PyTorch checkpoint (`.pt`)
- Содержит: `model_state_dict`, `head_state_dict`, `epoch`, `val_acc`
- Размер: 46 MB

---

## Установка

```bash
# 1. Создать виртуальное окружение
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt
```

**Требования:** Python 3.8+, желательно CUDA (но работает и на CPU)

---

## Как использовать

### Идентификация по одному фото

```bash
python src/identify_person.py --image data/raw/20/20_5.bmp
```

Что выводит: предсказанный ID человека, confidence (сходство 0-1), топ-5 кандидатов.

Аргументы:
```
--image     Путь к фото (ОБЯЗАТЕЛЬНО)
--model     Путь к весам (default: results/best/huita.pt)
--data      Папка с галереей (default: data/raw)
--device    cuda или cpu (default: cuda)
```

Пример с явными аргументами:
```bash
python src/identify_person.py \
    --image data/raw/30/30_10.bmp \
    --model results/best/huita.pt \
    --data data/raw \
    --device cpu
```

---

### Проверить точность на всём датасете

```bash
python src/test_accuracy.py --model results/best/huita.pt
```

Тестирует 2 фото каждого из 834 людей (1668 тестов), выводит Top-1 и Top-5 accuracy.

---

### Найти минимум фото для надёжной идентификации одного человека

```bash
python src/find_min_samples.py --person-id 20 --test-image 1
```

Аргументы:
```
--person-id     ID человека (1-834)
--test-image    Какое фото использовать для теста (1-10)
--model         Путь к весам (default: results/best/huita.pt)
```

---

### Массовый тест по случайным людям

```bash
python src/find_min_samples_mass.py --num-people 150
```

---

## Как загрузить модель в своём коде

```python
import torch
from src.model_triplet import GrayscaleResNet18Triplet

# Загрузка
model = GrayscaleResNet18Triplet(embedding_dim=512, dropout_rate=0.1)
checkpoint = torch.load('results/best/huita.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Получить эмбеддинг для изображения
# image_tensor — torch.Tensor shape (1, 1, 224, 224), значения [0, 1]
with torch.no_grad():
    embedding = model(image_tensor)  # shape: (1, 512), L2-нормализован

# Сравнить два эмбеддинга (косинусное сходство)
similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
# similarity близко к 1.0 → один человек
# similarity близко к 0.0 → разные люди
```

---

## Препроцессинг изображений (важно!)

Модель ожидает конкретный препроцессинг. Без него точность будет плохой:

```python
import cv2
import numpy as np
import torch

def preprocess_image(image_path):
    # Читать grayscale (поддержка кириллицы в пути)
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    # Ресайз
    img = cv2.resize(img, (224, 224))
    
    # CLAHE — контрастирование (обязательно!)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Нормализация [0, 255] → [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # (H, W) → (1, 1, H, W) для батча из 1 изображения
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    
    return tensor
```

---

## Датасет

- **Структура:** `data/raw/{ID}/{ID}_{номер_фото}.bmp`
- **Пример:** `data/raw/42/42_3.bmp` — 3-е фото человека №42
- **Формат файлов:** .bmp, grayscale, 224×224 пикселей
- **Количество:** 834 человека × 10 фото = 8340 изображений

---

## FAQ

**Q: Можно на CPU?**
A: Да, добавь `--device cpu`. Будет медленнее, но работает.

**Q: Confidence близко к 0.9 у всех — это нормально?**
A: Да. Модель всегда найдёт "ближайшего" человека, даже если изображение не из датасета. Смотри на margin (разницу между 1-м и 2-м) — если он маленький (<0.01), идентификация ненадёжна.

**Q: Как добавить своих людей?**
A: Нужно дообучить модель. Код обучения в `src/train.py`. Просто добавить фото в `data/raw/835/`, `836/` и т.д. и запустить обучение заново.

**Q: Какой порог (threshold) для верификации?**
A: Из тестов — confidence ~0.926 в среднем для правильного совпадения. Рекомендуемый порог: **>0.85** для "это тот человек", но нужно подбирать под конкретную задачу.
