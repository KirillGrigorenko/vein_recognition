# ТЗ: Модуль биометрии венозных ладоней для СКУД

## Контекст

Django-проект СКУД (система контроля и управления доступом).
Весь код в папке `app/services/`. Порт 8000. Python 3.11, Django 4.2.

Сейчас биометрия — **заглушка** в `app/services/face_stub.py`, которая
возвращает случайные числа вместо реального сравнения.

Задача: заменить заглушку на реальный модуль **распознавания по венозному
рисунку ладони**.

---

## Что уже есть в проекте (не трогать архитектуру)

### Модель BiometricData (`app/services/models.py`)
```python
class BiometricData(models.Model):
    employee   = models.OneToOneField(Employee, on_delete=models.CASCADE)
    face_hash  = models.TextField(blank=True, null=True)   # сейчас не используется
    palm_hash  = models.TextField(blank=True, null=True)   # сюда писать хэш ладони
    face_registered_at = models.DateTimeField(null=True)
    palm_registered_at = models.DateTimeField(null=True)   # сюда дату регистрации
    status     = models.CharField(max_length=20, default='active')
```

### Текущий файл-заглушка `app/services/face_stub.py`
```python
def detect_face(image_bytes: bytes, employee) -> int:
    # возвращает confidence 0–100
    ...

def get_face_hash(image_bytes: bytes) -> str:
    # возвращает строковый хэш изображения
    ...
```

### Как вызывается в `app/services/api.py` (POST /api/auth/login/face/)
```python
from .face_stub import detect_face, get_face_hash

confidence = detect_face(image_bytes, employee)
face_hash  = get_face_hash(image_bytes)
```

### МИВАР-движок (`mes/mes_app/engine.py`)
Получает `confidence` (0–100) и принимает решение allowed / warning / denied.
Порог: confidence >= 70 → allowed, 50–69 → warning, < 50 → denied.
**Менять engine.py не нужно.**

---

## Что нужно реализовать

### 1. Новый файл `app/services/palm_bio.py`

Заменяет `face_stub.py`. Должен экспортировать те же две функции
(чтобы не менять `api.py`), но работать с ладонями:

```python
def detect_face(image_bytes: bytes, employee) -> int:
    """
    Принимает bytes изображения ладони (JPEG/PNG, ИК-камера).
    Сравнивает с зарегистрированным хэшем employee.biometricdata.palm_hash.
    Возвращает confidence: int 0–100.
    Если palm_hash не зарегистрирован — возвращает 0.
    """

def get_face_hash(image_bytes: bytes) -> str:
    """
    Принимает bytes изображения ладони.
    Возвращает строковое представление вектора/хэша ладони
    (base64 или hex), который будет сохранён в BiometricData.palm_hash.
    """
```

> Названия функций сохранить `detect_face` / `get_face_hash` —
> это намеренно, чтобы не менять `api.py`.

### 2. Обновить импорт в `app/services/api.py`

Заменить одну строку:
```python
# было:
from .face_stub import detect_face, get_face_hash
# стало:
from .palm_bio import detect_face, get_face_hash
```

### 3. Регистрация ладони — обновить `app/services/api.py`

Endpoint `POST /api/biometric/register/` сейчас пишет в `face_hash` и
`face_registered_at`. Переключить на `palm_hash` и `palm_registered_at`.

```python
# было:
bio.face_hash = get_face_hash(image_bytes)
bio.face_registered_at = now()
# стало:
bio.palm_hash = get_face_hash(image_bytes)
bio.palm_registered_at = now()
```

---

## Требования к модели распознавания

- Вход: изображение ладони в ИК-диапазоне (или обычный снимок если ИК нет),
  байты JPEG/PNG произвольного разрешения.
- Предобработка: обрезка ROI ладони, нормализация, Gabor-фильтр или
  аналог для выделения венозного рисунка.
- Представление: вектор признаков или бинарный код (например, PalmCode).
- Сравнение: расстояние Хэмминга (для бинарных кодов) или косинусное
  сходство (для векторов). Порог подобрать на своих данных.
- Результат сравнения нормировать в диапазон 0–100:
  - 100 = точное совпадение
  - 0   = полное несовпадение / не зарегистрирован

### Рекомендуемый стек
- OpenCV (`cv2`) — предобработка и фильтрация
- NumPy — работа с векторами
- Опционально: `torch` / `onnxruntime` если используется нейросеть

Зависимости добавить в `app/requirements.txt`.

---

## Структура файла `palm_bio.py` (скелет)

```python
import cv2
import numpy as np
import base64
from typing import Optional

def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Декодирует байты, выделяет ROI ладони, нормализует."""
    ...

def _extract_features(img: np.ndarray) -> np.ndarray:
    """Gabor / LBP / CNN — возвращает вектор признаков."""
    ...

def _hash_to_str(features: np.ndarray) -> str:
    """Сериализует вектор в строку для хранения в БД."""
    return base64.b64encode(features.astype(np.float32).tobytes()).decode()

def _str_to_hash(s: str) -> np.ndarray:
    """Десериализует строку обратно в вектор."""
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)

def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Возвращает сходство 0.0–1.0."""
    ...

def detect_face(image_bytes: bytes, employee) -> int:
    try:
        bio = employee.biometricdata
        if not bio.palm_hash:
            return 0
        stored = _str_to_hash(bio.palm_hash)
        features = _extract_features(_preprocess(image_bytes))
        sim = _similarity(features, stored)
        return int(sim * 100)
    except Exception:
        return 0

def get_face_hash(image_bytes: bytes) -> str:
    features = _extract_features(_preprocess(image_bytes))
    return _hash_to_str(features)
```

---

## Что НЕ трогать

| Файл | Причина |
|------|---------|
| `mes/mes_app/engine.py` | МИВАР-движок, работает с confidence как есть |
| `app/services/views.py` | HTML-вьюхи, биометрии не касаются |
| `app/services/models.py` | Поля уже есть, миграции не нужны |
| `app/services/templates/` | UI менять не нужно |
| `app/services/urls.py` | Маршруты не меняются |

---

## Файлы для изучения перед началом

Прочитай эти файлы чтобы понять систему:

1. `app/services/face_stub.py` — текущая заглушка (что заменяем)
2. `app/services/api.py` — как вызывается биометрия (строки с `detect_face`)
3. `app/services/models.py` — модель `BiometricData`
4. `mes/mes_app/engine.py` — как используется confidence
5. `CLAUDE.md` — общий контекст проекта

---

## Итог: минимальный чеклист

- [ ] Создать `app/services/palm_bio.py` с функциями `detect_face` и `get_face_hash`
- [ ] Обновить импорт в `app/services/api.py` (1 строка)
- [ ] Переключить регистрацию на `palm_hash` / `palm_registered_at` в `api.py`
- [ ] Добавить зависимости в `app/requirements.txt`
- [ ] Проверить: `detect_face` возвращает 0–100, не бросает исключений
