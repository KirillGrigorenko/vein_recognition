"""
🧠 MODEL V3.0 - ResNet18 ДЛЯ TRIPLET LEARNING (GRAYSCALE)
- Адаптирован для 1 канала (grayscale)
- ResNet18 backbone с transfer learning
- ТОЛЬКО Embedding head (нет classification)
- Оптимизировано для Triplet Loss
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GrayscaleResNet18Triplet(nn.Module):
    """
    ResNet18 для метрического обучения с Triplet Loss
    Выдает эмбеддинги для биометрического распознавания вен
    """
    
    def __init__(self, embedding_dim=512, dropout_rate=0.3, device='cuda'):
        super(GrayscaleResNet18Triplet, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        # Загружаем ResNet18 (pretrained)
        resnet18 = models.resnet18(pretrained=True)
        
        # Адаптируем первый conv слой для 1 канала вместо 3
        original_conv = resnet18.conv1
        self.conv1 = nn.Conv2d(
            in_channels=1,  # ✅ 1 канал для grayscale
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Копируем веса, усредняя по RGB каналам
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        # Заменяем первый слой
        resnet18.conv1 = self.conv1
        
        # Заморозить первые слои для transfer learning
        for param in list(resnet18.parameters())[:-20]:
            param.requires_grad = False
        
        # Берем только backbone (без fc layer)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        
        # ResNet18 выдает 512 features
        self.feature_dim = 512
        
        # Embedding head (для метрического обучения)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов новых слоев"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, 224, 224) - GRAYSCALE
        
        Returns:
            embeddings: Embedding features (batch_size, embedding_dim)
        """
        # Extract features через ResNet backbone
        features = self.backbone(x)  # (batch_size, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 512)
        
        # Generate embeddings (метрические признаки)
        embeddings = self.embedding_head(features)  # (batch_size, embedding_dim)
        
        # ✅ L2 normalization для косинусного расстояния
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embeddings(self, x):
        """Получить эмбеддинги для распознавания"""
        with torch.no_grad():
            embeddings = self.forward(x)
        return embeddings


def create_model(embedding_dim=512, dropout_rate=0.3, device='cuda'):
    """
    Создать модель для распознавания вен (Triplet Learning)
    
    Args:
        embedding_dim: Размерность эмбеддингов
        dropout_rate: Вероятность dropout
        device: 'cuda' или 'cpu'
    
    Returns:
        model: Модель на указанном device
    """
    model = GrayscaleResNet18Triplet(
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        device=device
    )
    
    # Переместить модель на device
    model = model.to(device)
    
    return model


if __name__ == '__main__':
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing model on {device}...")
    
    model = create_model(embedding_dim=512, device=device)
    
    # Dummy grayscale input
    x = torch.randn(4, 1, 224, 224).to(device)
    embeddings = model(x)
    
    print(f"\n✅ Model Test:")
    print(f"   Input shape: {x.shape} (Grayscale 1 channel)")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embeddings norm: {embeddings.norm(dim=1).mean():.4f} (should be ~1.0 after L2)")
    print(f"   Device: {device}")
    print(f"\n✅ Model works correctly!")
