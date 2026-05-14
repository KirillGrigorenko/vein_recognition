"""
ИДЕНТИФИКАЦИЯ ЧЕЛОВЕКА ПО ФОТО
Просто дай путь к фото и она скажет кто это
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_triplet import create_model
from preprocess import preprocess_image

class Identifier:
    def __init__(self, model_path, device='cuda'):
        print(f"Loading model...")
        self.model = create_model(embedding_dim=512, dropout_rate=0.1, device=device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device).eval()
        self.device = device
        print(f"✓ Model loaded, Epoch {checkpoint.get('epoch')}, Acc {checkpoint.get('val_acc', 0)*100:.1f}%\n")

    def get_embedding(self, image_path):
        img_tensor = preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            emb = self.model(img_tensor)
        return emb.cpu().numpy()[0]
    
    def identify(self, query_image, data_dir):
        print(f"Query image: {query_image}")
        print(f"Building gallery from {data_dir}...")
        
        query_emb = self.get_embedding(query_image)
        
        gallery_emb = []
        gallery_ids = []

        for person_dir in tqdm(sorted(Path(data_dir).iterdir()), leave=False):
            if not person_dir.is_dir():
                continue
            person_id = int(person_dir.name)
            embs = [self.get_embedding(str(p)) for p in sorted(person_dir.glob('*.bmp'))]
            if embs:
                centroid = np.mean(embs, axis=0)
                centroid /= np.linalg.norm(centroid)
                gallery_emb.append(centroid)
                gallery_ids.append(person_id)

        gallery_emb = np.array(gallery_emb)
        gallery_ids = np.array(gallery_ids)

        sims = np.dot(gallery_emb, query_emb)
        top_idx = np.argsort(-sims)[:5]
        
        top_id = gallery_ids[top_idx[0]]
        top_conf = sims[top_idx[0]]
        margin = sims[top_idx[0]] - sims[top_idx[1]] if len(top_idx) > 1 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ THIS IS PERSON #{top_id}")
        print(f"   Confidence: {top_conf:.4f}")
        print(f"   Margin: {margin:.6f}")
        print(f"{'='*60}\n")
        
        print(f"Top-5:")
        for rank, idx in enumerate(top_idx[:5]):
            print(f"  #{rank+1}: Person {gallery_ids[idx]:3d} | {sims[idx]:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--model', default='results/best/huita.pt')
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    identifier = Identifier(args.model, device=args.device)
    identifier.identify(args.image, args.data)
