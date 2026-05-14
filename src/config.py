"""
⚙️ CONFIG.PY FIXED V2 - ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ

Для 834 классов с разделением 8/2 по фото:
- Learning Rate повышен еще больше (0.002)
- Batch size уменьшен для лучшей сходимости
- Epochs увеличены
- Weight decay снижен
"""

import random
import numpy as np
import torch
from pathlib import Path

# ====== DEVICE SETTINGS ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== TRAINING PARAMETERS ======
BATCH_SIZE = 64  # Уменьшено с 96 для лучшей сходимости
EPOCHS = 100  # Увеличено
LEARNING_RATE = 0.0002  # Может быть повышено до 0.0005 если не сходится
WEIGHT_DECAY = 1e-4  # Снижено
DROPOUT_RATE = 0.3  # Снижено
EMBEDDING_DIM = 512

# ====== LOSS PARAMETERS ======
TRIPLET_MARGIN = 0.25

# ====== SCHEDULER & EARLY STOPPING ======
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'
EARLY_STOPPING_PATIENCE = 40  # Увеличено для большего датасета

# ====== OPTIMIZATION ======
RANDOM_SEED = 42
GRADIENT_CLIP = 1.0
USE_MIXED_PRECISION = True
NUM_WORKERS = 4

# ====== PATHS ======
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODEL_DIR = PROJECT_ROOT / 'results' / 'models'
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ====== AUGMENTATION CONFIG ======
AUGMENTATION_CONFIG = {
    'rotation': 15,
    'brightness': 0.2,
    'contrast': 0.2,
    'elastic': True,
    'perspective': True
}

def get_config():
    """Возвращает словарь конфигурации"""
    return {
        'DEVICE': DEVICE,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'DROPOUT_RATE': DROPOUT_RATE,
        'EMBEDDING_DIM': EMBEDDING_DIM,
        'TRIPLET_MARGIN': TRIPLET_MARGIN,
        'SCHEDULER_TYPE': SCHEDULER_TYPE,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'RANDOM_SEED': RANDOM_SEED,
        'GRADIENT_CLIP': GRADIENT_CLIP,
        'NUM_WORKERS': NUM_WORKERS,
        'DATA_DIR': str(DATA_DIR),
        'MODEL_DIR': str(MODEL_DIR),
        'METRICS_DIR': str(METRICS_DIR),
        'AUGMENTATION_CONFIG': AUGMENTATION_CONFIG
    }

def set_random_seed(seed=RANDOM_SEED):
    """Устанавливает random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def print_config():
    """Выводит конфигурацию"""
    config = get_config()
    print("\n" + "="*80)
    print("⚙️ CONFIGURATION (OPTIMIZED FOR 834 CLASSES)")
    print("="*80)
    print(f"🖥️ Device: {config['DEVICE']}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f" VRAM: {props.total_memory / 1e9:.2f} GB")

    print(f"\n📊 TRAINING:")
    print(f" Batch Size: {config['BATCH_SIZE']}")
    print(f" Epochs: {config['EPOCHS']}")
    print(f" Learning Rate: {config['LEARNING_RATE']} (with 10x boost = {config['LEARNING_RATE']*10})")
    print(f" Weight Decay: {config['WEIGHT_DECAY']}")
    print(f" Dropout: {config['DROPOUT_RATE']}")
    print(f" Early Stopping: {config['EARLY_STOPPING_PATIENCE']} эпох")

    print(f"\n📂 PATHS:")
    print(f" Data: {config['DATA_DIR']} ({'✅' if Path(config['DATA_DIR']).exists() else '❌'})")
    print(f" Models: {config['MODEL_DIR']}")
    print(f" Metrics: {config['METRICS_DIR']}")
    print("="*80 + "\n")

if __name__ == '__main__':
    set_random_seed()
    print_config()
    print("✅ Configuration loaded successfully!")
