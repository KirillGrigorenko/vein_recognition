"""
🎯 ОПРЕДЕЛИТЬ МИНИМАЛЬНОЕ КОЛИЧЕСТВО ФОТО ДЛЯ 95%+ ACCURACY
Однопользовательский тест (ИСПРАВЛЕННАЯ ВЕРСИЯ - совместима с config.py)
"""
import torch
from pathlib import Path
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_triplet import create_model
from preprocess import preprocess_image

class MinSamplesFinder:
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
    
    def test(self, person_id, data_dir, test_image_num=1):
        person_dir = Path(data_dir) / str(person_id)
        
        if not person_dir.exists():
            print(f"❌ Person {person_id} not found in {data_dir}")
            return
        
        # Получить все фото
        images = sorted(person_dir.glob('*.bmp'))
        
        if len(images) < 2:
            print(f"❌ Not enough images for person {person_id}")
            return
        
        print(f"Testing person {person_id}")
        print(f"Total images: {len(images)}")
        print(f"Test image: {person_id}_{test_image_num}.bmp")
        print("=" * 70)
        
        # Тестовое фото
        test_img_path = person_dir / f"{person_id}_{test_image_num}.bmp"
        if not test_img_path.exists():
            print(f"❌ Test image not found: {test_img_path}")
            return
        
        test_emb_norm = self.get_embedding(test_img_path)
        
        # Получить галерею (все фото кроме тестового)
        gallery_images = [img for img in images if img != test_img_path]
        
        print(f"\n{'N Samples':<12} {'Accuracy':<12} {'Confidence':<15} {'Margin':<15}")
        print("-" * 70)
        
        # Тест с разным количеством фото
        results = []
        for n_samples in range(1, len(gallery_images) + 1):
            gallery_emb = []
            gallery_ids = []
            
            # Выбрать первые n фото для галереи
            for img_path in gallery_images[:n_samples]:
                emb = self.get_embedding(img_path)
                gallery_emb.append(emb)
                gallery_ids.append(person_id)
            
            gallery_emb = np.array(gallery_emb)
            gallery_norm = gallery_emb / np.linalg.norm(gallery_emb, axis=1, keepdims=True)
            
            # Косинус similarity
            sims = np.dot(gallery_norm, test_emb_norm)
            top_idx = np.argsort(-sims)
            
            # Проверка
            top_ids = [gallery_ids[idx] for idx in top_idx]
            correct = (top_ids[0] == person_id)
            confidence = sims[top_idx[0]]
            margin = sims[top_idx[0]] - sims[top_idx[1]] if len(top_idx) > 1 else 0
            
            status = "✓" if correct else "✗"
            accuracy = 100 if correct else 0
            
            print(f"{n_samples:<12} {accuracy:>6.1f}%{'':<4} {confidence:>8.6f}{'':<6} {margin:>8.6f}")
            
            results.append({
                'n_samples': n_samples,
                'correct': correct,
                'confidence': confidence,
                'margin': margin
            })
        
        # Рекомендация
        print("\n" + "=" * 70)
        print("RECOMMENDATION:")
        print("=" * 70)
        
        # Найти минимум для 100% accuracy
        correct_results = [r for r in results if r['correct']]
        
        if correct_results:
            min_samples = correct_results[0]['n_samples']
            min_result = correct_results[0]
            print(f"\n✓ Minimum {min_samples} photo(s) for 100% accuracy on this person")
            print(f"  - Confidence: {min_result['confidence']:.6f}")
            print(f"  - Margin: {min_result['margin']:.6f}")
        else:
            print(f"\n✗ Even with all photos, not correctly identified!")
        
        # Информация о margin
        print(f"\n💡 Информация о margin (разница top-1 и top-2):")
        for r in results[:5]:
            if r['margin'] > 0.01:
                print(f"   {r['n_samples']} samples: margin = {r['margin']:.6f} (отличный margin!)")
            elif r['margin'] > 0.001:
                print(f"   {r['n_samples']} samples: margin = {r['margin']:.6f} (хороший margin)")
            else:
                print(f"   {r['n_samples']} samples: margin = {r['margin']:.6f} (слабый margin)")

def main():
    parser = argparse.ArgumentParser(description="Find min photos for 95%+ accuracy (single person)")
    parser.add_argument('--person-id', type=int, required=True, help='Person ID to test')
    parser.add_argument('--test-image', type=int, default=1, help='Which image to test (1-10)')
    parser.add_argument('--model', default='results/best/huita.pt', help='Model path')
    parser.add_argument('--data', default='data/raw', help='Data directory')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    finder = MinSamplesFinder(args.model, device=args.device)
    finder.test(args.person_id, args.data, args.test_image)

if __name__ == '__main__':
    main()
