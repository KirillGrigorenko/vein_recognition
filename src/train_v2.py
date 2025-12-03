"""
🚀 TRAIN V2.0 - ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ И ОПТИМИЗИРОВАННЫЙ
- ✅ Правильные импорты (lr_scheduler, config_v2)
- ✅ Батчи правильно распаковываются (словарь)
- ✅ CPU оптимизация
- ✅ Аргумент --epochs переопределяет EPOCHS ДО вывода конфига!
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

import config_v2
from model_v2 import create_model
from vein_data_loader import VeinDatasetManager


# ✅ ПЕРЕОПРЕДЕЛЯЕМ ДО import! Это главное!
def parse_args():
    """Парсим аргументы ДО всего остального"""
    parser = argparse.ArgumentParser(description='Train Vein Recognition Model V2')
    parser.add_argument('--root-dir', type=str, default='data/raw', help='Root directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='Workers for DataLoader')
    return parser.parse_args()


# Парсим аргументы в самом начале
args = parse_args()

# ✅ ПЕРЕОПРЕДЕЛЯЕМ ПЕРЕМЕННЫЕ В МОДУЛЕ config_v2!
if args.epochs is not None:
    config_v2.EPOCHS = args.epochs

if args.batch_size is not None:
    config_v2.BATCH_SIZE = args.batch_size

if args.lr is not None:
    config_v2.LEARNING_RATE = args.lr

# Теперь импортируем всё остальное
from config_v2 import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, 
    EMBEDDING_DIM, DEVICE, RANDOM_SEED, MODELS_DIR, METRICS_DIR, 
    EARLY_STOPPING_PATIENCE, USE_SCHEDULER, SCHEDULER_TYPE, GRADIENT_CLIP,
    set_random_seed, print_config, check_files_exist
)


class EarlyStopping:
    """Early Stopping с контролем по validation loss"""

    def __init__(self, patience=15, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"   ✅ Улучшение! Val Loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"   ⚠️  Без улучшений {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"   🛑 EARLY STOPPING! Best Epoch: {self.best_epoch}, Loss: {self.best_loss:.4f}")


class VeinTrainer:
    """Trainer для модели распознавания вен"""

    def __init__(self, model, train_loader, val_loader, num_epochs, device='cpu', verbose=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose

        # Optimizer - Adam с weight decay
        self.optimizer = Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # Scheduler
        if USE_SCHEDULER:
            if SCHEDULER_TYPE == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.num_epochs,
                    eta_min=1e-7
                )
            else:  # plateau
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=False
                )
        else:
            self.scheduler = None

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=1e-4,
            verbose=True
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.best_model_path = None

    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch [{epoch+1}/{self.num_epochs}] Train",
            leave=False
        )

        for batch_idx, batch in pbar:
            # ✅ ПРАВИЛЬНО: батч это словарь!
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            logits, _ = self.model(images, labels)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = 100.0 * total_correct / total_samples
            pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{avg_acc:.2f}%')

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * total_correct / total_samples

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Валидация"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Validating",
                leave=False
            )

            for batch in pbar:
                # ✅ ПРАВИЛЬНО: батч это словарь!
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                logits, _ = self.model(images, labels)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100.0 * total_correct / total_samples

        return epoch_loss, epoch_acc

    def train(self):
        """Главный цикл обучения"""
        print("\n" + "="*80)
        print("🚀 НАЧАЛО ОБУЧЕНИЯ (V2.0 - CPU OPTIMIZED)")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Weight Decay (L2): {WEIGHT_DECAY}")
        print(f"Dropout Rate: {DROPOUT_RATE}")
        print(f"Embedding Dim: {EMBEDDING_DIM}")
        print(f"Scheduler: {SCHEDULER_TYPE if USE_SCHEDULER else 'None'}")
        print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
        print(f"Total Epochs: {self.num_epochs}")
        print("="*80 + "\n")

        for epoch in range(self.num_epochs):
            # Train
            print(f"📊 Epoch [{epoch+1}/{self.num_epochs}]")
            train_loss, train_acc = self.train_epoch(epoch)

            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

            # Validate
            val_loss, val_acc = self.validate_epoch()
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")

            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = MODELS_DIR / f'model_best_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"  ✅ Сохранена лучшая модель: {self.best_model_path.name}")

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            self.early_stopping(val_loss, epoch)
            if self.early_stopping.early_stop:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

        print("\n" + "="*80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("="*80 + "\n")

        self.save_history()
        self.plot_history()

    def save_history(self):
        """Сохранение истории"""
        history_path = METRICS_DIR / 'training_history_v2.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"📊 История сохранена: {history_path}")

    def plot_history(self):
        """Построение графиков"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss vs Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy vs Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plot_path = METRICS_DIR / 'training_curves_v2.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Графики сохранены: {plot_path}")


def main():
    # Config
    set_random_seed(RANDOM_SEED)
    print_config()  # Теперь печатает ПРАВИЛЬНЫЕ значения!

    if not check_files_exist():
        print("❌ Missing required files!")
        return

    # Dataset
    print("\n📂 Загрузка датасета...")
    manager = VeinDatasetManager(args.root_dir)
    manager.analyze()

    train_indices, val_indices = manager.create_train_val_split(train_size=0.8)

    train_loader, val_loader = manager.create_dataloaders(
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        augmentation=True
    )

    # Model
    print("\n🧠 Создание модели...")
    model = create_model(
        num_classes=834,
        embedding_dim=EMBEDDING_DIM,
        dropout_rate=DROPOUT_RATE,
        device=DEVICE,
        use_mobilenet=True
    )

    # Trainer
    trainer = VeinTrainer(
        model,
        train_loader,
        val_loader,
        num_epochs=EPOCHS,
        device=DEVICE
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
