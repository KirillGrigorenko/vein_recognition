"""
⚙️ CONFIG V2.0 - ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ ДЛЯ CPU
"""

import os
import random
import numpy as np
import torch
from pathlib import Path


# ====== DEVICE SETTINGS ======
DEVICE = 'cpu'  # CPU only

# ====== TRAINING PARAMETERS ======
BATCH_SIZE = 16  # ✅ Меньше для CPU (было 32)
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 regularization
DROPOUT_RATE = 0.2  # ✅ Меньше dropout для CPU (было 0.3)
EMBEDDING_DIM = 256  # ✅ Меньше embeddings (было 512)

# ====== SCHEDULER & EARLY STOPPING ======
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # 'cosine' or 'plateau'
EARLY_STOPPING_PATIENCE = 15

# ====== OPTIMIZATION ======
RANDOM_SEED = 42
GRADIENT_CLIP = 1.0
USE_MIXED_PRECISION = False  # Не поддерживается на CPU хорошо

# ====== PATHS ======
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'

# Создай директории если их нет
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ====== AUGMENTATION CONFIG ======
AUGMENTATION_CONFIG = {
    'rotation': 15,
    'brightness': 0.2,
    'contrast': 0.2,
    'elastic': True,
    'perspective': False  # ✅ Отключи для CPU (тяжело)
}


def set_random_seed(seed=42):
    """Устанавливает random seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Детерминированные вычисления
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config():
    """Выводит конфигурацию"""
    print("\n" + "="*80)
    print("⚙️  КОНФИГУРАЦИЯ ПРОЕКТА V2.0 (ОПТИМИЗИРОВАНО ДЛЯ CPU)")
    print("="*80)
    print(f"🖥️  Device: {DEVICE}")
    print(f"🎲 Random Seed: {RANDOM_SEED}")
    
    print(f"\n📊 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Weight Decay (L2): {WEIGHT_DECAY}")
    print(f"   Dropout Rate: {DROPOUT_RATE}")
    print(f"   Embedding Dim: {EMBEDDING_DIM}")
    print(f"   Scheduler: {SCHEDULER_TYPE if USE_SCHEDULER else 'None'}")
    print(f"   Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   Gradient Clip: {GRADIENT_CLIP}")
    
    print(f"\n📂 ПУТИ:")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Raw data: {DATA_DIR} ({'✅' if DATA_DIR.exists() else '❌'})")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Metrics: {METRICS_DIR}")
    
    print(f"\n🎨 ПАРАМЕТРЫ АУГМЕНТАЦИИ:")
    for key, val in AUGMENTATION_CONFIG.items():
        print(f"   {key}: {val}")
    
    print("="*80 + "\n")


def check_files_exist():
    """Проверяет наличие необходимых файлов"""
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return False
    
    if not list(DATA_DIR.glob('*')):
        print(f"❌ Data directory is empty: {DATA_DIR}")
        return False
    
    return True
