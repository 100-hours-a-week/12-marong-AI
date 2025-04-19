import torch
import torch.nn.functional as F
import numpy as np

class GetReviewEmbedding:
    def __init__(self, embedding_model, min_length=5):
        self.embedding_model = embedding_model
        self.min_length = min_length

    def embedding(self, review_texts):
        vectors = [
            self.embedding_model.encode(text, convert_to_tensor=True)
            for text in review_texts
            if isinstance(text, str) and len(text.strip()) > self.min_length
        ]

        if not vectors:
            # 임베딩 모델 차원 유추 (기본 768)
            try:
                dummy = self.embedding_model.encode("예시 문장", convert_to_tensor=True)
                dim = dummy.shape[-1]
            except:
                dim = 768
            return np.zeros((1, dim))

        stacked = torch.stack(vectors)  # (N, dim)
        avg = stacked.mean(dim=0, keepdim=True)
        return F.normalize(avg, dim=1).cpu().numpy()