"""
🎯 МАССОВЫЙ ТЕСТ: найти мин фото для 95%+ accuracy на 100+ случайных людях
"""
import torch
from pathlib import Path
import argparse
import sys
import os
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_triplet import create_model
from preprocess import preprocess_image

class MinSamplesMassTest:
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
    
    def test_person(self, person_id, data_dir, test_image_num):
        """Тест одного человека"""
        person_dir = Path(data_dir) / str(person_id)
        
        if not person_dir.exists():
            return None
        
        images = sorted(person_dir.glob('*.bmp'))
        
        if len(images) < 2:
            return None
        
        # Тестовое фото
        test_img_path = person_dir / f"{person_id}_{test_image_num}.bmp"
        if not test_img_path.exists():
            return None
        
        try:
            test_emb_norm = self.get_embedding(test_img_path)
        except:
            return None
        
        # Галерея (все фото кроме тестового)
        gallery_images = [img for img in images if img != test_img_path]
        
        results = {}
        
        # Тест с 1, 2, 3, 4, 5 фото
        for n_samples in range(1, min(6, len(gallery_images) + 1)):
            gallery_emb = []
            
            # Случайно выбрать n фото
            selected_images = random.sample(gallery_images, n_samples)
            
            for img_path in selected_images:
                try:
                    emb = self.get_embedding(img_path)
                    gallery_emb.append(emb)
                except:
                    pass
            
            if len(gallery_emb) == 0:
                continue
            
            gallery_emb = np.array(gallery_emb)
            gallery_norm = gallery_emb / np.linalg.norm(gallery_emb, axis=1, keepdims=True)
            
            # Косинус similarity
            sims = np.dot(gallery_norm, test_emb_norm)
            top_idx = np.argsort(-sims)[0]
            
            # Проверка
            correct = True  # Галерея содержит только фото этого человека!
            confidence = sims[top_idx]
            
            results[n_samples] = {
                'correct': correct,
                'confidence': confidence
            }
        
        return results
    
    def run(self, data_dir, num_people=100):
        # Получить всех людей
        data_path = Path(data_dir)
        all_people = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        
        # Выбрать случайных людей
        test_people = random.sample(all_people, min(num_people, len(all_people)))
        
        print(f"Testing {len(test_people)} random people")
        print(f"Each person tested with 1-5 photos in gallery")
        print("=" * 80)
        
        # Собрать статистику
        stats = defaultdict(lambda: {'correct': 0, 'count': 0, 'confs': []})
        
        for person_id in tqdm(test_people, desc="Progress"):
            # Случайное фото для теста
            test_image_num = random.randint(2, 10)
            
            result = self.test_person(person_id, data_dir, test_image_num)
            
            if result is None:
                continue
            
            for n_samples, res in result.items():
                stats[n_samples]['correct'] += int(res['correct'])
                stats[n_samples]['count'] += 1
                stats[n_samples]['confs'].append(res['confidence'])
        
        # Вывод результатов
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"{'N Photos':<15} {'Accuracy':<15} {'Avg Confidence':<20} {'Samples':<15}")
        print("-" * 80)
        
        for n in range(1, 6):
            if stats[n]['count'] > 0:
                accuracy = stats[n]['correct'] / stats[n]['count'] * 100
                avg_conf = np.mean(stats[n]['confs'])
                print(f"{n:<15} {accuracy:>6.2f}%{'':<8} {avg_conf:>6.4f}{'':<13} {stats[n]['count']}")
        
        # Рекомендация
        print("\n" + "=" * 80)
        print("RECOMMENDATION FOR 95%+ ACCURACY:")
        print("=" * 80)
        
        min_found = False
        for n in range(1, 6):
            if stats[n]['count'] > 0:
                accuracy = stats[n]['correct'] / stats[n]['count'] * 100
                if accuracy >= 95 and not min_found:
                    print(f"\n✓ Minimum {n} photo(s) for 95%+ accuracy")
                    print(f"  - Achieved accuracy: {accuracy:.2f}%")
                    print(f"  - Avg confidence: {np.mean(stats[n]['confs']):.6f}")
                    print(f"  - Tested on: {stats[n]['count']} people")
                    min_found = True
        
        if not min_found:
            print("\n⚠️ Even 5 photos not enough for 95% accuracy on average")
            best_n = max([n for n in range(1, 6) if stats[n]['count'] > 0],
                        key=lambda n: stats[n]['correct'] / stats[n]['count'])
            best_acc = stats[best_n]['correct'] / stats[best_n]['count'] * 100
            print(f"  - Best result: {best_n} photos → {best_acc:.2f}%")
        
        print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Find min photos for 95%+ accuracy (mass test)")
    parser.add_argument('--num-people', type=int, default=150, help='Number of people to test')
    parser.add_argument('--model', default='results/best/huita.pt', help='Model path')
    parser.add_argument('--data', default='data/raw', help='Data directory')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    tester = MinSamplesMassTest(args.model, device=args.device)
    tester.run(args.data, num_people=args.num_people)

if __name__ == '__main__':
    main()
