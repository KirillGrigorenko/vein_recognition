"""
Модуль для загрузки датасета палмовых вен с улучшенным preprocessing
- CLAHE для нормализации контраста
- ✅ ПОЛНЫЙ FIX: Поддержка Cyrillic путей на Windows через cv2.imdecode
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class PalmVeinDatasetFromFolder(Dataset):
    """
    Dataset для работы с иерархией папок + улучшенный preprocessing
    ✅ Поддержка Cyrillic путей на Windows через numpy.fromfile + cv2.imdecode
    """
    
    def __init__(self,
                 root_dir='raw',
                 person_indices=None,
                 person_id_map=None,
                 augmentation=False,
                 target_size=(224, 224),
                 enhance_contrast=True):
        self.root_dir = Path(root_dir).resolve()
        self.target_size = target_size
        self.augmentation = augmentation
        self.person_id_map = person_id_map or {}
        self.enhance_contrast = enhance_contrast
        
        # Найти все папки людей
        all_person_dirs = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])
        
        # Если маппинг не передан, создать его
        if not self.person_id_map:
            self.person_id_map = {
                person_dir: idx
                for idx, person_dir in enumerate(all_person_dirs)
            }
        
        # Соберем все (путь к изображению, класс/человек)
        self.image_paths = []
        self.labels = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.BMP', '.JPG', '.JPEG', '.PNG'}
        
        for person_dir in all_person_dirs:
            person_path = self.root_dir / person_dir
            person_label = self.person_id_map[person_dir]
            
            # Если задан фильтр индексов, пропустить других
            if person_indices is not None and person_label not in person_indices:
                continue
            
            # Найти все изображения в папке человека
            for img_path in person_path.iterdir():
                if img_path.suffix.lower() in valid_extensions and img_path.is_file():
                    self.image_paths.append(img_path)
                    self.labels.append(person_label)
        
        print(f"✓ Загружено датасета:")
        print(f" - Изображений: {len(self.image_paths)}")
        print(f" - Уникальных людей: {len(set(self.labels))}")
        print(f" - Аугментация: {'ДА' if augmentation else 'НЕТ'}")
        print(f" - Улучшение контраста: {'ДА (CLAHE)' if enhance_contrast else 'НЕТ'}")
    
    def __len__(self):
        return len(self.image_paths)
    
    @staticmethod
    def _enhance_contrast(img_array):
        """
        Применить CLAHE (Contrast Limited Adaptive Histogram Equalization)
        для нормализации контраста
        """
        # CLAHE работает с uint8
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Создаем CLAHE объект
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_array)
        
        return img_enhanced
    
    @staticmethod
    def _read_image_safe(img_path):
        """
        ✅ Безопасное чтение изображения на Windows с Cyrillic путями
        Использует numpy.fromfile + cv2.imdecode вместо cv2.imread
        """
        try:
            # ✅ Способ 1: cv2.imdecode + numpy.fromfile (РАБОТАЕТ с Cyrillic!)
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                return img
        except Exception as e:
            pass
        
        try:
            # Способ 2: PIL fallback
            img_pil = Image.open(img_path).convert("L")
            img = np.array(img_pil, dtype=np.uint8)
            return img
        except Exception as e:
            print(f"⚠️ Не удалось загрузить: {img_path} ({e})")
            return None
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # ✅ Безопасное чтение с поддержкой Cyrillic
        img = self._read_image_safe(img_path)
        
        if img is None:
            img = np.zeros(self.target_size, dtype=np.uint8)
        else:
            # Resize
            img = cv2.resize(img, self.target_size)
            
            # ✅ УЛУЧШЕНИЕ КОНТРАСТА
            if self.enhance_contrast:
                img = self._enhance_contrast(img)
        
        # Применить аугментацию если нужна
        if self.augmentation:
            img = self._augment(img)
        
        # Нормализировать (0-255 -> 0-1)
        img = img.astype(np.float32) / 255.0
        
        # ✅ Оставляем 1 канал: (H, W) -> (1, H, W)
        img_tensor = np.expand_dims(img, axis=0)
        
        return {
            'image': torch.from_numpy(img_tensor).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'path': str(img_path)
        }
    
    def _augment(self, img):
        """Простая аугментация для изображений ладоней"""
        return img


class VeinDatasetManager:
    """
    Менеджер для управления датасетом
    """
    
    def __init__(self, root_dir='raw', enhance_contrast=True):
        self.root_dir = Path(root_dir).resolve()
        self.person_id_map = {}
        self.all_person_dirs = []
        self.stats = {}
        self.enhance_contrast = enhance_contrast
        self._build_person_map()
    
    def _build_person_map(self):
        """Построить маппинг людей -> индексы"""
        self.all_person_dirs = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])
        
        self.person_id_map = {
            person_dir: idx
            for idx, person_dir in enumerate(self.all_person_dirs)
        }
        
        print(f"✓ Найдено людей: {len(self.all_person_dirs)}")
    
    def analyze(self):
        """Анализ структуры датасета"""
        stats = {
            'total_people': len(self.all_person_dirs),
            'total_images': 0,
            'images_per_person': [],
            'min_images': float('inf'),
            'max_images': 0,
        }
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.BMP', '.JPG', '.JPEG', '.PNG'}
        
        for person_dir in self.all_person_dirs:
            person_path = self.root_dir / person_dir
            images = [
                f.name for f in person_path.iterdir()
                if f.suffix.lower() in {ext.lower() for ext in valid_extensions} and f.is_file()
            ]
            
            num_images = len(images)
            stats['total_images'] += num_images
            stats['images_per_person'].append(num_images)
            stats['min_images'] = min(stats['min_images'], num_images)
            stats['max_images'] = max(stats['max_images'], num_images)
        
        avg_images = np.mean(stats['images_per_person']) if stats['images_per_person'] else 0
        self.stats = stats
        
        print("\n" + "="*60)
        print("📊 АНАЛИЗ ДАТАСЕТА")
        print("="*60)
        print(f"✓ Люди (субъекты): {stats['total_people']}")
        print(f"✓ Всего изображений: {stats['total_images']}")
        print(f"✓ Среднее на человека: {avg_images:.1f}")
        print(f"✓ Минимум/Максимум: {stats['min_images']}/{stats['max_images']}")
        print("="*60 + "\n")
        
        return stats
    
    def create_train_val_split(self,
                               train_size=0.8,
                               random_state=42):
        """
        Создать train/val split на уровне людей
        """
        all_indices = list(range(len(self.all_person_dirs)))
        
        train_indices, val_indices = train_test_split(
            all_indices,
            train_size=train_size,
            random_state=random_state
        )
        
        print(f"\n✓ Split создан:")
        print(f" - Train: {len(train_indices)} людей")
        print(f" - Val: {len(val_indices)} людей")
        print(f" - Ratio: {len(train_indices)}/{len(val_indices)} = {train_size:.1%}")
        
        return train_indices, val_indices
    
    def create_dataloaders(self,
                           train_indices,
                           val_indices,
                           batch_size=32,
                           num_workers=4,
                           augmentation=True):
        """
        Создать DataLoaders для train и val
        """
        train_dataset = PalmVeinDatasetFromFolder(
            root_dir=str(self.root_dir),
            person_indices=train_indices,
            person_id_map=self.person_id_map,
            augmentation=augmentation,
            target_size=(224, 224),
            enhance_contrast=self.enhance_contrast
        )
        
        val_dataset = PalmVeinDatasetFromFolder(
            root_dir=str(self.root_dir),
            person_indices=val_indices,
            person_id_map=self.person_id_map,
            augmentation=False,
            target_size=(224, 224),
            enhance_contrast=self.enhance_contrast
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\n✓ DataLoaders созданы:")
        print(f" - Train batches: {len(train_loader)}")
        print(f" - Val batches: {len(val_loader)}")
        
        return train_loader, val_loader


if __name__ == '__main__':
    print("🚀 Инициализация системы загрузки датасета (V3 - Cyrillic Fix FINAL)\n")
    
    manager = VeinDatasetManager(root_dir='data/raw', enhance_contrast=True)
    stats = manager.analyze()
    
    train_idx, val_idx = manager.create_train_val_split(train_size=0.8)
    
    train_loader, val_loader = manager.create_dataloaders(
        train_indices=train_idx,
        val_indices=val_idx,
        batch_size=32,
        augmentation=True
    )
    
    print("\n✓ Проверка первого батча из train_loader:")
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        print(f" - Shape: {images.shape} (Grayscale 1 channel)")
        print(f" - Labels: {labels[:5]}... (первые 5)")
        print(f" - Dtype: {images.dtype}")
        print(f" - Min/Max: {images.min():.4f} / {images.max():.4f}")
        break
    
    print("\n✅ Датасет готов к обучению!")
