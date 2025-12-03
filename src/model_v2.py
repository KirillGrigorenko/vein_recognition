"""
🧠 MODEL V2 - ОПТИМИЗИРОВАННЫЙ ДЛЯ CPU
- MobileNetV2 backbone (легче ResNet18)
- Инициализация весов Xavier
- Без нормализации (убивает сигнал)
- Эффективные FC слои
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet18


def create_model(num_classes=834, embedding_dim=256, dropout_rate=0.2, device='cpu', use_mobilenet=True):
    """Создаёт оптимизированную модель"""
    
    model = VeinFeatureExtractor(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        use_mobilenet=use_mobilenet
    )
    
    model = model.to(device)
    return model


class VeinFeatureExtractor(nn.Module):
    """Оптимизированная модель для распознавания вен - CPU friendly"""
    
    def __init__(self, num_classes=834, embedding_dim=256, dropout_rate=0.2, use_mobilenet=True):
        super(VeinFeatureExtractor, self).__init__()
        
        if use_mobilenet:
            # ✅ MobileNetV2 - ЛЕГЧЕ для CPU
            backbone = mobilenet_v2(weights=None)  # Без pretrained весов для CPU
            
            # Адаптируем первый слой для grayscale
            original_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                1, 32, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                bias=False
            )
            with torch.no_grad():
                backbone.features[0][0].weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
            
            self.backbone = backbone.features
            backbone_out_dim = 1280
        else:
            # ✅ ResNet18 - если нужна точность
            backbone = resnet18(weights=None)
            
            # Адаптируем первый слой для grayscale
            original_conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                1, 64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            with torch.no_grad():
                backbone.conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)
            
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_out_dim = 512
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Embedding FC layers - эффективно
        self.fc1 = nn.Linear(backbone_out_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights - правильная инициализация
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x, label=None):
        """
        Args:
            x: Input tensor (batch, 1, 224, 224)
            label: Labels (batch,) - для совместимости
        
        Returns:
            logits: (batch, num_classes)
            embeddings: (batch, embedding_dim)
        """
        # Backbone
        feat = self.backbone(x)
        
        # Global Average Pooling
        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)
        
        # Embedding FC
        embeddings = self.fc1(feat)
        embeddings = self.bn1(embeddings)
        embeddings = self.relu(embeddings)
        
        if self.training:
            embeddings = self.dropout(embeddings)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits, embeddings
