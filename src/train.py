"""
🚀 TRAIN AGGRESSIVE V3 - ОЧЕНЬ ВЫСОКИЙ LEARNING RATE

Problem: Loss не падает даже с 0.002 LR
Solution: Increase LR to 0.01-0.02 (ОЧЕНЬ АГРЕССИВНО!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import sys
import os
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_triplet import create_model
from config import get_config, set_random_seed, print_config

# ==================== DATA LOADER ====================

class PalmVeinDatasetSplit8_2(Dataset):
    """Dataset с разделением 8/2 ПО ФОТО"""
    def __init__(self, root_dir='raw', split='train', target_size=(224, 224), enhance_contrast=True):
        self.root_dir = Path(root_dir).resolve()
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast
        self.split = split

        self.all_person_dirs = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        self.person_id_map = {
            person_dir: idx
            for idx, person_dir in enumerate(self.all_person_dirs)
        }

        self.image_paths = []
        self.labels = []

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.BMP', '.JPG', '.JPEG', '.PNG'}

        for person_dir in self.all_person_dirs:
            person_path = self.root_dir / person_dir
            person_label = self.person_id_map[person_dir]

            img_list = sorted([
                f for f in person_path.iterdir()
                if f.suffix.lower() in valid_extensions and f.is_file()
            ])

            if split == 'train':
                selected_images = img_list[:8]
            else:
                selected_images = img_list[8:]

            for img_path in selected_images:
                self.image_paths.append(img_path)
                self.labels.append(person_label)

        print(f"✓ Dataset ({split}): {len(self.image_paths)} images, {len(set(self.labels))} people")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _enhance_contrast(img_array):
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_array)

    @staticmethod
    def _read_image_safe(img_path):
        try:
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        except:
            pass

        try:
            img_pil = Image.open(img_path).convert("L")
            img = np.array(img_pil, dtype=np.uint8)
            return img
        except:
            return None

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = self._read_image_safe(img_path)

        if img is None:
            img = np.zeros(self.target_size, dtype=np.uint8)
        else:
            img = cv2.resize(img, self.target_size)

            if self.enhance_contrast:
                img = self._enhance_contrast(img)

        img = img.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img, axis=0)

        return {
            'image': torch.from_numpy(img_tensor).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'path': str(img_path)
        }


# ==================== CLASSIFICATION HEAD ====================

class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=834):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        # Инициализируем веса
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, embeddings):
        return self.fc(embeddings)


# ==================== TRAINER ====================

class TrainerAggressive:
    def __init__(self, config, num_classes, device='cuda'):
        self.config = config
        self.device = device
        self.num_classes = num_classes

        # Model
        self.model = create_model(
            embedding_dim=config['EMBEDDING_DIM'],
            dropout_rate=config['DROPOUT_RATE'],
            device=device
        )
        self.model.to(device)

        # Classification head
        self.head = ClassificationHead(
            embedding_dim=config['EMBEDDING_DIM'],
            num_classes=num_classes
        ).to(device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer - ОЧЕНЬ ВЫСОКИЙ LR!
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.head.parameters()),
            lr=0.01,  # ОЧЕНЬ ВЫСОКИЙ! (было 0.002)
            weight_decay=0,  # ОТКЛЮЧЕН! (было 1e-4)
            betas=(0.9, 0.999)
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

        self.history = {
            'train_loss': [],
            'val_acc': [],
            'best_epoch': 0,
            'best_acc': 0
        }

        print(f"\n⚙️  TRAINER CONFIG:")
        print(f"  Learning Rate: 0.01 (ОЧЕНЬ ВЫСОКИЙ!)")
        print(f"  Weight Decay: 0 (ОТКЛЮЧЕН!)")
        print(f"  Optimizer: Adam")

    def train_epoch(self, train_loader):
        self.model.train()
        self.head.train()

        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            embeddings = self.model(images)
            logits = self.head(embeddings)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.head.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / max(batch_count, 1)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        self.head.eval()

        all_predictions = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            embeddings = self.model(images)
            logits = self.head(embeddings)

            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy = np.sum(all_predictions == all_labels) / len(all_labels) * 100

        return accuracy

    def train(self, train_loader, val_loader):
        print("\n" + "="*80)
        print("🚀 TRAINING - AGGRESSIVE V3 (HIGH LR = 0.01)")
        print("="*80)
        print(f"Number of classes: {self.num_classes}")
        print(f"Train images: {len(train_loader) * self.config['BATCH_SIZE']}")
        print(f"Test images: {len(val_loader) * self.config['BATCH_SIZE']}")
        print("="*80 + "\n")

        no_improvement = 0

        for epoch in range(self.config['EPOCHS']):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            val_acc = self.validate(val_loader)
            self.history['val_acc'].append(val_acc)

            print(f"\n📊 Epoch [{epoch+1}/{self.config['EPOCHS']}]")
            print(f" Train Loss: {train_loss:.4f}")
            print(f" Val Acc@1: {val_acc:.2f}%")
            print(f" LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > self.history['best_acc']:
                self.history['best_acc'] = val_acc
                self.history['best_epoch'] = epoch
                no_improvement = 0

                model_path = Path(self.config['MODEL_DIR']) / f"model_best_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'head_state_dict': self.head.state_dict(),
                    'val_acc': val_acc,
                }, model_path)

                print(f" ✅ BEST! Saved: {model_path.name}")
            else:
                no_improvement += 1
                print(f" ⚠️  No improvement {no_improvement}/{self.config['EARLY_STOPPING_PATIENCE']}")

            # Scheduler step
            self.scheduler.step(val_acc)

            if no_improvement >= self.config['EARLY_STOPPING_PATIENCE']:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

        # Save history
        history_path = Path(self.config['METRICS_DIR']) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n✅ TRAINING COMPLETED")
        print(f"Best epoch: {self.history['best_epoch'] + 1}")
        print(f"Best accuracy: {self.history['best_acc']:.2f}%")


# ==================== MAIN ====================

def main():
    set_random_seed()
    print_config()

    config = get_config()
    device = config['DEVICE']

    print("\n📂 Loading dataset with 8/2 split...")

    train_dataset = PalmVeinDatasetSplit8_2(
        root_dir=config['DATA_DIR'],
        split='train',
        target_size=(224, 224),
        enhance_contrast=True
    )

    test_dataset = PalmVeinDatasetSplit8_2(
        root_dir=config['DATA_DIR'],
        split='test',
        target_size=(224, 224),
        enhance_contrast=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )

    print(f"\n✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Test loader: {len(val_loader)} batches")

    num_classes = len(train_dataset.person_id_map)
    print(f"\n✓ Total classes: {num_classes}")

    trainer = TrainerAggressive(config, num_classes=num_classes, device=device)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
