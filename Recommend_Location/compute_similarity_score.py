from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import torch.nn.functional as F

class ComputeSimilarityScore:
    def __init__(self, model):
        self.model = model

    def compute(self, mbti_vector, review_vector):
        if review_vector is None:
            return 0.0  # 또는 None

        # 🧠 1. MBTI 예측 → vibe 벡터
        mbti_tensor = torch.tensor(mbti_vector, dtype=torch.float32)
        if mbti_tensor.ndim == 1:
            mbti_tensor = mbti_tensor.unsqueeze(0)

        with torch.no_grad():
            pred_vibe = self.model(mbti_tensor)
            pred_vibe = F.normalize(pred_vibe, dim=1).cpu().numpy()  # (1, 768)

        # 🧠 2. review_vector 타입/차원 보정
        if isinstance(review_vector, torch.Tensor):
            review_vector = review_vector.detach().cpu().numpy()
        elif isinstance(review_vector, list):
            review_vector = np.array(review_vector)

        # 🛠️ review_vector가 3차원이면 → squeeze
        while review_vector.ndim > 2:
            review_vector = np.squeeze(review_vector, axis=0)

        # 🛠️ review_vector가 1차원이면 → reshape
        if review_vector.ndim == 1:
            review_vector = review_vector.reshape(1, -1)

        # ✅ cosine similarity 계산
        return cosine_similarity(pred_vibe, review_vector)[0][0]