"""
ТЕСТ ТОЧНОСТИ - используем EMBEDDINGS (как в identify_person.py)
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

class Tester:
    def __init__(self, model_path, device='cuda'):
        print(f"Loading model...")
        self.model = create_model(embedding_dim=512, dropout_rate=0.1, device=device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device).eval()
        self.device = device
        print(f"✓ Model loaded\n")

    def get_embedding(self, image_path):
        img_tensor = preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            emb = self.model(img_tensor)
        return emb.cpu().numpy()[0]

    def test(self, data_dir):
        print(f"Loading dataset from {data_dir}...")
        data_path = Path(data_dir)
        all_people = sorted([d for d in data_path.iterdir() if d.is_dir()])

        print(f"Total people: {len(all_people)}")
        print(f"Split: 8 train + 2 test photos per person\n")

        print(f"Building centroid gallery (mean of 8 embeddings per person)...")
        gallery_emb = []
        gallery_ids = []

        for person_dir in tqdm(all_people, leave=False):
            person_id = int(person_dir.name)
            images = sorted(person_dir.glob('*.bmp'))

            embs = [self.get_embedding(str(p)) for p in images[:8]]
            centroid = np.mean(embs, axis=0)
            centroid /= np.linalg.norm(centroid)
            gallery_emb.append(centroid)
            gallery_ids.append(person_id)

        gallery_emb = np.array(gallery_emb)
        gallery_ids = np.array(gallery_ids)

        print(f"Gallery: {len(gallery_emb)} centroids (one per person)\n")

        print(f"Testing (2 photos per person)...")
        top1_correct = 0
        top5_correct = 0
        confs = []
        margins = []
        total_test = 0

        for person_dir in tqdm(all_people, leave=False):
            person_id = int(person_dir.name)
            images = sorted(person_dir.glob('*.bmp'))

            for img_path in images[8:]:
                total_test += 1
                query_emb = self.get_embedding(str(img_path))

                sims = np.dot(gallery_emb, query_emb)
                top_idx = np.argsort(-sims)[:5]

                top_ids = gallery_ids[top_idx]
                top_sims = sims[top_idx]

                if person_id == top_ids[0]:
                    top1_correct += 1
                if person_id in top_ids[:5]:
                    top5_correct += 1

                confs.append(top_sims[0])
                margins.append(top_sims[0] - top_sims[1] if len(top_sims) > 1 else 0)

        confs = np.array(confs)
        margins = np.array(margins)

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Top-1: {top1_correct/total_test*100:.2f}% ({top1_correct}/{total_test})")
        print(f"Top-5: {top5_correct/total_test*100:.2f}% ({top5_correct}/{total_test})")
        print(f"\nConfidence: {confs.mean():.4f} ± {confs.std():.4f}")
        print(f"Margin: {margins.mean():.6f} ± {margins.std():.6f}")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/best/huita.pt')
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    tester = Tester(args.model, device=args.device)
    tester.test(args.data)
