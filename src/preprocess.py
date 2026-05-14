import cv2
import numpy as np
import torch
from pathlib import Path


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Единый препроцессинг для train и inference.
    Порядок: grayscale → resize → CLAHE → normalize → tensor.
    """
    img_array = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        from PIL import Image
        img = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)

    img = cv2.resize(img, target_size)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
